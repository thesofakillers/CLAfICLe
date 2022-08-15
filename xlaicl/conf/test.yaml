defaults:
  - _self_
  - benchmark: test
  - trainer: test
  # https://stackoverflow.com/a/70777327/9889508
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

model_name: "metaicl"
checkpoint_path: "checkpoints/metaicl/model.pt"
en: true
de: true
fr: true

# to avoid hydra creating output dirs: https://stackoverflow.com/a/64635492/9889508
hydra:
  run:
    dir: .
  output_subdir: null
