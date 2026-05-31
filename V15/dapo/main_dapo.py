#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2025/06/17 19:17:35
@Author  :   wangjiakang
@File    :   main_dapo.py
'''


import hydra
import ray

from .dapo_ray_trainer_v8 import RayDAPOTrainer


def get_custom_reward_fn(config):
    from rewards.general_reward import general_reward_fn

    return general_reward_fn


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)

        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        from verl.workers.reward_manager import get_reward_manager_cls

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        reward_manager_cls = get_reward_manager_cls(reward_manager_name)

        compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        dynamic_method = config.trainer.get("dynamic_sampling", {}).get("method", "default")
        if dynamic_method == "full_dataset_v11":
            from .dapo_ray_trainer_v11 import RayDAPOTrainerV11

            print("[V11] Using RayDAPOTrainerV11 with full_dataset_v11 method")
            trainer = RayDAPOTrainerV11(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                device_name=config.trainer.device,
            )
        elif dynamic_method == "sample_taylor_v13":
            from .dapo_ray_trainer_v13 import RayDAPOTrainerV13

            print("[V13] Using RayDAPOTrainerV13 with sample_taylor_v13 method")
            trainer = RayDAPOTrainerV13(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                device_name=config.trainer.device,
            )
        else:
            print("[V8] Using RayDAPOTrainer")
            trainer = RayDAPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                device_name=config.trainer.device,
            )

        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
