import os
import numpy as np

from LyMARL.basestation import SmallCellBaseStation
from LyMARL.user_equipment import UserEquipment
from LyMARL.core import generate_triangle_coverage

from LyMARL.utils_mappo import set_seed
from LyMARL.env_mappo import MAPPOEnvironment
from LyMARL.trainer_mappo import MAPPOTrainer


if __name__ == "__main__":
    seed = 0
    set_seed(seed)

    area_size = 100
    num_users = 20

    sbs_positions = generate_triangle_coverage(area_size, 35)
    sbs_list = [SmallCellBaseStation(i + 1, pos, 10, 35) for i, pos in enumerate(sbs_positions)]
    users = [
        UserEquipment(i + 1, (np.random.uniform(10, 90), np.random.uniform(10, 90)))
        for i in range(num_users)
    ]

    # --------------------------------------------------
    # Training environment
    # --------------------------------------------------
    train_env = MAPPOEnvironment(
        base_stations=sbs_list,
        users=users,
        V=5.0,
        power_budget_ratio=0.6,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=100,
        bs_top_k=5,
        hard_window_len=10000,
        bs_over_penalty=100.0,
        use_hard_constraint=False,   # training: no hard constraint
    )

    trainer = MAPPOTrainer(
        env=train_env,
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

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    train_npz_path = "LyMARL_train_ep100_queue_reset.npz"
    model_path = "LyMARL_ep100.pt"

    trainer.train(
        n_episodes=100,
        steps_per_episode=10000,
        update_interval=128,
        queue_reset_interval=10000,  
        save_npz_path=train_npz_path,
    )

    trainer.save_model(model_path)

    print(f"\n✅ Training rewards saved to: {os.path.abspath(train_npz_path)}")
    print(f"✅ Model saved to: {os.path.abspath(model_path)}")

    # --------------------------------------------------
    #enable hard constraint only for eval
    # --------------------------------------------------
    trainer.env.set_hard_constraint(True)

    eval_npz_path = "LyMARL_eval_ep100_queue_reset.npz"
    trainer.evaluate(
        n_episodes=1, 
        steps_per_episode=10000,
        queue_reset_interval=10000,
        save_npz_path=eval_npz_path
    )

    print("\n✅ Completed!\n")