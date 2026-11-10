# Third-party detector files

The evaluation scores decoded RADIal cubes with **FFTRadNet** (Rebut et al., CVPR 2022), frozen, with its official weights. The official-test run also scores some reconstructions with **AdaRadar's retrained 4-bit FFTRadNet** (Park, Chun and Seok, CVPR 2026). These files belong to their authors and are not committed here:

| File the code loads | Source | Needed by |
|---|---|---|
| `FFTRadNet.py` | `FFTRadNet/model/FFTRadNet.py` in https://github.com/valeoai/RADIal | `tools/radial_ppo_tuned_gpu`, `tools/radial_official_test_gpu` |
| `config_FFTRadNet_192_56.json` | `FFTRadNet/config/config_FFTRadNet_192_56.json` in the same repository | `tools/radial_ppo_tuned_gpu`, `tools/radial_official_test_gpu` |
| `fftradnet_state_dict.pt` | made from `FFTRadNet_RA_192_56_epoch78_loss_172.8239_AP_0.9813.pth`, linked as "pre-trained models" in `FFTRadNet/README.md` of the RADIal repository | `tools/radial_ppo_tuned_gpu`, `tools/radial_official_test_gpu` |
| `adaradar_4bit_state_dict.pt` | made from `FFTRadNet_pretrained_4bit.pth`, linked as "pretrained weights" in the README of https://github.com/jp4327/adaradar | `tools/radial_official_test_gpu` only |

The RADIal repository has no license file, so we do not redistribute its code or weights. AdaRadar's repository is Apache-2.0. We link to its weights instead of copying them, so the provenance stays with the authors.

## Placing the files

```bash
python scripts/setup_third_party.py \
    --radial-repo /path/to/RADIal \
    --fftradnet-checkpoint /path/to/FFTRadNet_RA_192_56_epoch78_loss_172.8239_AP_0.9813.pth \
    --adaradar-checkpoint /path/to/FFTRadNet_pretrained_4bit.pth      # needed by the official-test run only
```

What the script does:

1. **Checks the downloads** against the SHA-256 hashes of the files used in the paper runs:

   | File | SHA-256 |
   |---|---|
   | `FFTRadNet.py` | `be8dcaee5924d3122cbeea73981baaf6881292993ae5e3289d1a2687d4d4839c` |
   | `config_FFTRadNet_192_56.json` | `a710a9af2b6c948655f07d94d97cd9e164fc1b02466139021384890b854f3089` |
   | `FFTRadNet_RA_192_56_epoch78_loss_172.8239_AP_0.9813.pth` | `d9545e7383b29193243ef08b245f633b64b9c24d7097ccc96a3d8a98f76dcdd9` |
   | `FFTRadNet_pretrained_4bit.pth` | `080ee74cdfb6ee0906daf845320cc671c51c1e14b510bb83a5249f8d5e1ee256` |

2. **Converts each checkpoint to a weights-only state dict.** It opens each checkpoint with `torch.load(weights_only=True)` and allows only a few numpy types that the training history stores, so no pickled code runs. It keeps `net_state_dict` only. The official RADIal checkpoint stores its tensors under older module names than the released `FFTRadNet.py`, so three prefixes are renamed: `backbone.preproc.` → `FPN.pre_enc.`, `backbone.` → `FPN.`, `RAmap_header.` → `RA_decoder.`. No tensor is changed, and the renamed names must equal the class's parameter names exactly.
3. **Verifies the tensors.** It computes a hash over the name, dtype, shape and bytes of every tensor and compares it with the tensors used in the paper runs (`bcc0c066…` for FFTRadNet, `91871524…` for AdaRadar). File bytes can differ between PyTorch versions, so the tensors are compared, not the files.
4. **Copies** the four files into every package folder listed in `scripts/third_party_slots.json`. `.gitignore` keeps them out of commits.

If a hash differs, the script stops. `--force` continues anyway, but the results may then differ from the paper.

## AdaRadar compressor

The AdaRadar baseline does not need AdaRadar's code at run time. `radar_sim/radial_v18_adaradar.py` re-implements its static compressor: a 64×64 block DCT, top-k pruning and 4-bit scaled quantization. Its feedback controller is in `tools/run_radial_v18_followup.py`. Both follow the official code (`utils/dct.py`, `utils/evaluation_dct_0th_order_optimize.py`). On the repository's sample `fft_000018.npy`, the re-implementation matches the official `dct_based_compression` to 1e-7 relative RMS error. AdaRadar is released under Apache-2.0.
