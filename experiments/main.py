# experiments/main.py

import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  
sys.path.insert(0, ROOT)

import numpy as np

from basestation import SmallCellBaseStation
from user_equipment import UserEquipment
from core import generate_triangle_coverage

from LyMARL.env import MAPPOEnvironment
from LyMARL.trainer import MAPPOTrainer


if __name__ == "__main__":
    area_size = 100
    num_users = 20

    sbs_positions = generate_triangle_coverage(area_size, 35)
    sbs_list = [SmallCellBaseStation(i + 1, pos, 10, 35) for i, pos in enumerate(sbs_positions)]
    users = [UserEquipment(i + 1, (np.random.uniform(10, 90), np.random.uniform(10, 90)))
             for i in range(num_users)]

    env = MAPPOEnvironment(
        base_stations=sbs_list,
        users=users,
        V=5.0,
        power_budget_ratio=0.6,
        enable_mobility=True,
        enable_channel_variation=True,
        lambda_p=2.0,
        on_window=100,
        bs_top_k=5,
        hard_window_len=10000,
        bs_over_penalty=100.0,
    )

    trainer = MAPPOTrainer(
        env=env,
        lr_actor_ue=3e-4,
        lr_actor_bs=3e-4,
        lr_critic_ue=1e-3,
        lr_critic_bs=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef_ue=0.05,
        entropy_coef_bs=0.05,
        value_coef_ue=0.5,
        value_coef_bs=0.5,
        n_epochs=4,
        minibatch_size=256
    )

    train_steps = 50000
    train_results = trainer.train(n_steps=train_steps, update_interval=128)
    trainer.plot_results(train_results, tag=f"train_{train_steps//1000}k")

    eval_steps = 100000
    eval_results = trainer.evaluate(n_steps=eval_steps)
    trainer.plot_results(eval_results, tag=f"eval_{eval_steps//1000}k")

    print("\n✅ Completed!\n")