Here's the plan:

cnn-occlusion-robustness/
  src/
    data/
      gtsrb.py                    # dataset + split
      augment.py                  # uses camera_occlusion.transforms
    models/
      simple_cnn.py
      resnet.py
    train.py                      # training loop
    eval.py                       # per-condition eval
    build_eval_matrix.py          # writes evaluation_matrix_YYYYMMDD.csv
    analysis/
      advanced_analysis_report.py # rename from your current analysis file
  configs/
    base.yaml
    train/
      clean.yaml
      rain_light.yaml
      dust_heavy.yaml
    eval/
      matrix.yaml                 # test grid of effects
  scripts/
    run_all.sh
    reproduce_paper.sh
  results/                        # .gitignore
  artifacts/                      # selected CSV/figures for the README
  reports/
  figures/
  README.md
  requirements.txt / pyproject.toml
  LICENSE

