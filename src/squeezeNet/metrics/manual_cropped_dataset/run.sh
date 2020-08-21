set -x
python join_dataset_results.py   val_dataset.csv    val_probabilities.csv       -o val_results_squeezenet.csv
python join_dataset_results.py   test_dataset.csv   test_probabilities.csv      -o test_results_squeezenet.csv
python join_dataset_results.py   test_dataset.csv   test_probabilities.csv      -o test_results_squeezenet.csv
python join_dataset_results.py   dadosTreino/trainig_dataset.csv   dadosTreino/probabilities.csv    -o train_results_squeezenet.csv 
python metrics.py   -i val_results_squeezenet.csv   -o val_metrics.txt          -r val_roc_curve.txt          -p val_pr_curve.txt  > val_log.txt
python metrics.py   -i test_results_squeezenet.csv  -o test_metrics.txt         -r test_roc_curve.txt         -p test_pr_curve.txt        -tr val_roc_curve.txt         -tp val_pr_curve.txt  > test_log.txt
python metrics.py   -i val_results_squeezenet.csv   -o val_metrics_0.0025.txt   -r val_roc_curve_0.0025.txt   -p val_pr_curve_0.0025.txt  -s 0.0025  > val_log_0.0025.txt
python metrics.py   -i test_results_squeezenet.csv  -o test_metrics_0.0025.txt  -r test_roc_curve_0.0025.txt  -p test_pr_curve_0.0025.txt -tr val_roc_curve_0.0025.txt  -tp val_pr_curve_0.0025.txt  -s 0.0025  > test_log_0.0025.txt
python metrics.py   -i val_results_squeezenet.csv   -o val_metrics_0.001.txt    -r val_roc_curve_0.001.txt    -p val_pr_curve_0.001.txt   -s 0.001   > val_log_0.001.txt
python metrics.py   -i test_results_squeezenet.csv  -o test_metrics_0.001.txt   -r test_roc_curve_0.001.txt   -p test_pr_curve_0.001.txt  -tr val_roc_curve_0.001.txt   -tp val_pr_curve_0.001.txt   -s 0.001   > test_log_0.001.txt

python metrics.py   -i train_results_squeezenet.csv -o train_metrics.txt        -r train_roc_curve.txt        -p train_pr_curve.txt       > train_log.txt

gnuplot run.gp
set +x
