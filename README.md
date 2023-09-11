# AS_TSF

# Update
  .
# Environment
 Pytorch 1.12.1

 Numpy 1.22.4

## Datasets
  You can download all datasets from https://drive.google.com/drive/folders/1I819ARskzUCvDS76f8OqLuVaVFqsZ0LK?usp=sharing.

  The downloaded folders 'Fund_all' and 'Fund_all_2' should be placed at the 'dataset' folder. Then you can following the annotation in 'run_baselines_fund.py' and run it to reproduce the paper results. 'VS' denotes Vanilla scale scaling. 'SC' denotes only using our scale calibrating sub-module. 'SS+SC' is equivalent to out AS module, using both our scale calibrating and scaling selection sub-modules.