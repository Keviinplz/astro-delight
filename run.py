import os
import socket
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from delight.training.cnn import train_delight_cnn_model
from delight.training.dataset import DelightDatasetOptions
from delight.notifier.telegram import TelegramNotifier
from functools import partial
from datetime import datetime

def run_ray_tune(*, name: str, num_samples: int, gpus_per_trial: float, source: str):   
    params = {
        "nconv1": tune.lograndint(16,  64 + 1),
        "nconv2": tune.lograndint(16,  64 + 1),
        "nconv3": tune.lograndint(16,  64 + 1),
        "ndense": tune.lograndint(256,  2048 + 1),
        "dropout": tune.uniform(0, 0.4),
        "batch_size": tune.lograndint(16,  64 + 1),
        "lr": tune.loguniform(1e-4, 1e-2),
        "epochs": 50
    } 
    
    options = DelightDatasetOptions(
        source=source,
        n_levels=5,
        fold=0,
        mask=False,
        object=True,
        rot=True,
        flip=True
    )

    scheduler = ASHAScheduler(
        grace_period=20, # epochs before evaluate early stop
        reduction_factor=3, # the worst 1/3 trials will be terminated 
        brackets=1 # we don't want to decrease resources
    )


    train_fn = partial(train_delight_cnn_model, options=options)
    
    tuner = tune.Tuner(
        tune.with_resources(train_fn, resources={"gpu": gpus_per_trial}), #type: ignore
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples
        ),
        run_config=train.RunConfig(
            name=name
        ),
        param_space=params
    )
    return tuner.fit()

def main():
    now = datetime.now()
    name = f"ray_experiment_{now.strftime('%d_%m_%Y-%H_%M_%S')}"
    num_samples = 200
    machine = socket.gethostname()
    chat_id = -4049822363
    token = "6333721085:AAGbLdRmJsn8TU-gTrSu8npgXNgOaNmBwcs"
    notifier = TelegramNotifier(token=token, chat_id=chat_id)
    sources = {
        "quimal-gpu.alerce.online": "/home/kpinochet/astro-delight/data",
        "LAPTOP-CUH9J3SR": "/home/keviinplz/universidad/tesis/snhost/data"
        }

    default_source = "/home/keviinplz/universidad/tesis/snhost/data"

    message = f"""
    **Experimento `{name}` iniciado el día {now.strftime('%d-%m-%Y a las %H:%M:%S')} UTC**

    Información del experimento:

    ```
    Pruebas: {num_samples}
    Máquina: {socket.gethostname()}
    ```
    """

    notifier.notify(message)
    
    result = run_ray_tune(
        name=name, 
        num_samples=num_samples, 
        gpus_per_trial=0.2, 
        source=sources.get(machine, default_source)
    )

    finish = datetime.now()

    df = result.get_dataframe()

    df_folder = os.path.join(os.getcwd(), 'ray_results_df')
    os.makedirs(df_folder, exist_ok=True)

    df_filename = os.path.join(df_folder, name + ".pkl") 
    result_path = '/'.join(result.get_best_result().path.split("/")[:-1])
    df.to_pickle(df_filename)

    best_quantity = 10
    data = df.sort_values(by=["val_loss"])[["val_loss", "train_loss"]].head(best_quantity).to_dict(orient="records") # type: ignore

    rows = ""
    for i, d in enumerate(data):
        rows +=f'    |   {i+1}  |  {round(d["val_loss"],3)}  |    {round(d["train_loss"],3)}   |\n'

    message = f"""
    **El experimento `{name}` ha finalizado**

    Mejores {len(data)} resultados:

    ```
    | Rank | val_loss | train_loss |
    |------|:--------:|:----------:|
    {rows}
    ```

    Este experimento se ha ejecutado en la máquina {socket.gethostname()}
    Y fue iniciado el día {now.strftime('%d-%m-%Y a las %H:%M:%S')} UTC
    Finalizando el día {finish.strftime('%d-%m-%Y a las %H:%M:%S')} UTC

    Se ha guardado un dataframe con los resultados en `{df_filename}`

    A su vez, el experimento se encuentra en `{result_path}`
    """
    notifier.notify(message)

if __name__ == "__main__":
    main()
