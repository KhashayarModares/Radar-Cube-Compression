# Datasets

Neither dataset is redistributed here. Download them from their authors and pass their folders to the scripts with `--dataset-root`. All RADIal scripts also take `--reference-root`, a clone of the RADIal repository, whose signal-processing code and calibration table the receiver uses.

## RADIal (raw sequences)

*Rebut et al., "Raw High-Definition Radar for Multi-Task Learning", CVPR 2022.* Repository and download links: https://github.com/valeoai/RADIal

1. Clone the repository. This is `--reference-root`:
   ```bash
   git clone https://github.com/valeoai/RADIal.git
   ```
   The code reads `SignalProcessing/rpl.py` and `SignalProcessing/CalibrationTable.npy` from it. `scripts/setup_third_party.py` also takes `FFTRadNet/model/FFTRadNet.py` and `FFTRadNet/config/config_FFTRadNet_192_56.json` from it.
2. Download the **raw sequences** (not only the ready-to-use subset) and `labels_CVPR.csv` from the link in the RADIal README. Arrange them as:
   ```
   <dataset-root>/
   ├── labels_CVPR.csv
   ├── RECORD@2020-11-21_11.54.31/
   │   ├── RECORD@2020-11-21_11.54.31_radar_ch0.bin   … _radar_ch3.bin
   │   ├── RECORD@2020-11-21_11.54.31_camera.mjpg      (only for the qualitative figures)
   │   └── …
   └── RECORD@…/
   ```
   The scripts decode the four radar chips into cubes of 16 receive channels × 256 chirps × 512 samples, pair them with the labels at a 20 ms synchronization tolerance, and check the result against the frozen split before any cube is used.

**Splits.** The split files are frozen and versioned with the code:

| Use | File | Train | Validation | Test |
|---|---|---|---|---|
| Training, selection, test on 662 cubes | `tools/radial_ppo_tuned_gpu/GLOBAL_SPLIT_V17.csv` | 2,074 cubes, 18 recordings | 708 cubes, 11 recordings | 662 cubes, 7 recordings |
| Official test | `tools/radial_official_test_gpu/GLOBAL_SPLIT_OFFICIAL.csv` | 2,074 | 528 | 2,048 = 662 + 1,386 cubes of RADIal's 4 official test recordings |

The official split adds the three official test recordings that the first split did not use, and moves recording `2020-11-22_12.03.47` from validation to test. That recording helped select the PPO checkpoint, which the paper states. Each split has a `*.provenance.json` file with its SHA-256 and cube counts per recording; the runners refuse a split that differs from it.

## UWCR, UW raw-ADC automotive dataset

*Gao et al., "RAMP-CNN: A Novel Neural Network for Enhanced Automotive Radar Object Recognition", IEEE Sensors Journal, 2021.* Repository and download links: https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection

TI AWR1843, 2 Tx (TDM) × 4 Rx = 8 virtual channels, 255 chirps × 128 samples per frame, 22 sequences, 19,754 frames. Arrange the sequences as:

```
<dataset-root>/
├── 2019_04_09_bms1000/
│   ├── radar_raw_frame/000000.mat …     raw ADC frames (adcData, 128 × 255 × 4 × 2)
│   ├── text_labels/0000000000.csv …     per-frame labels [uid, class, px, py, wid, len]
│   └── images_0/                        camera frames (not needed)
└── …
```

Labels give position, not Doppler, so a label is matched in range and angle, with the angle estimated by an FFT over the virtual channels. Some sequences write label fields as floats, which the loaders accept.

**Splits.** 14 training, 4 validation and 4 test sequences, split by sequence. The cube lists are in `results/runs/uw_ppo_*/tables/cube_index_*.csv`, and the selection rule is in the README.

| Role | Sequences |
|---|---|
| Test | `2019_04_09_cms1000`, `2019_04_30_pbms002`, `2019_05_09_mlms003`, `2019_05_29_mlms006` |
| Validation | `2019_04_09_pms3000`, `2019_04_30_mlms001`, `2019_05_09_pbms004`, `2019_05_29_bcms000` |
| Train | the other 14 |
