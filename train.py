import torch
import torch.nn.functional as F
import fire
from mistral_sae.sae import SparseAutoencoder
from mistral_sae.activationsLoader import ActivationsLoader
from mistral_sae.config import get_model_config

def train(model: str, data_folder: str):
    """
    Training script for Sparse Autoencoder using the specified model.

    Args:
        model_name (str): Name of the model to train on (e.g., 'mistral-7b', 'pixtral-12b').
    """
    # Get model configuration
    config = get_model_config(model)
    if config is None:
        print(f"Model '{model}' not found in configurations.")
        return

    # Set model parameters based on configuration
    D_MODEL = config.hidden_size
    D_HIDDEN = config.vocab_size
    BATCH_SIZE = 1024  # Adjust as needed
    scale = D_HIDDEN / (2**14)
    lr = 2e-4 / scale**0.5

    # Initialize model
    sae = SparseAutoencoder(D_MODEL, D_HIDDEN)
    sae = sae.to("cuda")
    optimizer = torch.optim.AdamW(
        sae.parameters(), lr=lr, eps=6.25e-10, betas=(0.9, 0.999)
    )

    MISTRAL_MODEL_PATH = config.local_path
    actsLoader = ActivationsLoader(128, 512, MISTRAL_MODEL_PATH, target_layer=16, zst_folder_name=data_folder, act_dir=f"{model}-act", d_model=D_MODEL)

    def loss_fn(x, recons, auxk):
        mse_scale = 1.0 / 19.9776  # Adjust based on your data
        auxk_coeff = 1.0 / 32.0

        mse_loss = mse_scale * F.mse_loss(recons, x)
        if auxk is not None:
            auxk_loss = auxk_coeff * F.mse_loss(auxk, x - recons).nan_to_num(0)
        else:
            auxk_loss = torch.tensor(0.0)
        return mse_loss, auxk_loss

    scaler = torch.cuda.amp.GradScaler()
    count = 0
    while True:
        new_batch = actsLoader.new_data().to("cuda")
        single_batches = torch.split(new_batch, BATCH_SIZE)
        for batch in single_batches:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                recons, auxk, num_dead = sae(batch)
                mse_loss, auxk_loss = loss_fn(batch, recons, auxk)
                loss = mse_loss + auxk_loss

            scaler.scale(loss).backward()

            sae.norm_weights()
            sae.norm_grad()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if count % 1000 == 0:
                # Periodically save model
                torch.save(sae.state_dict(), f"{model}.pth")

            if count % 50 == 0:
                # Logging (you can add your logging code here)
                pass

            count += 1

        if actsLoader.needs_refresh():
            # Move model off GPU to make room
            sae = sae.cpu()
            optimizer.zero_grad(set_to_none=True)
            scaler_state = scaler.state_dict()

            torch.cuda.empty_cache()
            actsLoader.refresh_data()

            # Move everything back onto the GPU
            scaler = torch.cuda.amp.GradScaler()
            scaler.load_state_dict(scaler_state)
            sae = sae.to("cuda")
            optimizer = torch.optim.AdamW(
                sae.parameters(), lr=lr, eps=6.25e-10, betas=(0.9, 0.999)
            )

if __name__ == "__main__":
    fire.Fire(train)
