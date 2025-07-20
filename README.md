# Diffoot - Graph-Conditioned Diffusion Model for Predicting Football Player Movements (Modifying)
Defense team player trajectory prediction in soccer


# Requirements
In requirements.txt


# Get the data
Download the raw data [here](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177)


# Reference github
[idsse-data](https://github.com/spoho-datascience/idsse-data?tab=readme-ov-file)

[LauireOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)

[Metrica Sports](https://github.com/metrica-sports/sample-data)

```    
    SoccerTrajPredict/
                │
                ├── idsse-data/                           # raw_data file directory
                │   ├── DFL-MAT-J03WMX
                │   │      ├──DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml
                │   │      ├──DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml
                │   │      └──DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml
                │   ├── DFL-MAT-J03WN1
                |   └── ... (for all 7 matchs)
                ├── codes/ 
                │   ├── utils/                          # Util codes
                │   │      ├── data_utils.py            # utils for data processing
                |   |      ├── graph_utils.py           # utils for building graph data
                │   │      ├── data_processing.py       # processing tools from idsse-data
                │   │      ├── Metrica_EPV.py           # Tools from LauireOnTracking
                │   │      ├── Metrica_IO.py            
                │   │      ├── Metrica_PitchControl.py            
                │   │      ├── Metrica_Velocities.py            
                │   │      ├── Metrica_Viz.py       
                │   │      └── utils.py                 # essential tools from references
                │   ├── models/                         # Model codes
                │   │      ├── diff_model.py            # Diffusion Model
                |   |      ├── diff_modules.py          # Diffusion Modules (Denoising network)
                |   |      ├── encoder.py               # Encoder model codes
                |   |      ├── pretrainig_encoder.py    # pretraing encoder for conditions
                │   │      ├── lstm_model.py            # Time-Series Model for comparing
                │   │      └── transformer_model.py
                │   │
                │   ├── make_dataset.py             # Generating Dataset
                |   ├── main_for_diffusion.py       # Main.py for diffusion model
                |   └── main_for_timeseries.py      # Main.py for time-series model
                │
                ├── requirements.txt            # Dependencies
                |
                └── README.md                   # Project documentation
```
