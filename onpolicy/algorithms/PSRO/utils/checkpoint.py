import os
import shutil

import torch


def restore_eval_policy(policy, model_dir, head_str=None):
    """Restore policy actor parameters from a saved checkpoint head."""
    if head_str is None:
        actor_path = os.path.join(str(model_dir), "actor.pt")
    else:
        actor_path = os.path.join(str(model_dir), "actor_" + str(head_str) + ".pt")

    policy_actor_state_dict = torch.load(actor_path)
    policy.actor.load_state_dict(policy_actor_state_dict)


def transfer_policy_A2B_full(model_dir, head_str_prev, head_str_new):
    """Copy actor, critic, and value-normalizer checkpoints between checkpoint heads."""
    source_path = os.path.join(model_dir, "actor_" + str(head_str_prev) + ".pt")
    destination_path = os.path.join(model_dir, "actor_" + str(head_str_new) + ".pt")
    shutil.copy(source_path, destination_path)
    print(f"Copied actor from {source_path} to {destination_path}")

    source_path = os.path.join(model_dir, "critic_" + str(head_str_prev) + ".pt")
    if os.path.isfile(source_path):
        destination_path = os.path.join(model_dir, "critic_" + str(head_str_new) + ".pt")
        shutil.copy(source_path, destination_path)
        print(f"Copied critic from {source_path} to {destination_path}")

    source_path = os.path.join(model_dir, "vnorm_" + str(head_str_prev) + ".pt")
    if os.path.isfile(source_path):
        destination_path = os.path.join(model_dir, "vnorm_" + str(head_str_new) + ".pt")
        shutil.copy(source_path, destination_path)
        print(f"Copied value normalizer from {source_path} to {destination_path}")
