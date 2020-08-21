set -x

python ../join_dataset_results.py   cropped_test_clean_list.csv   probabilities_clean.csv   -o testClean_fullMammograms_squeezenet.csv

time \
python ../metrics.py  	-i testClean_fullMammograms_squeezenet.csv  \
			-o test_clean_metrics.txt \
			-r test_clean_roc_curve.txt  \
			-p test_clean_pr_curve.txt \
			-tr ../val_roc_curve.txt \
		        -tp ../val_pr_curve.txt  \
			> test_clean_log.txt

cat test_log.txt

gnuplot <<< " \
	set terminal png size 800,600; \
	set output 'test_clean_best_roc_curve.png'; \
	set xlabel '1 - Specificity'; \
	set ylabel 'Sensitivity'; \
	plot [0:1][0:1] 'test_clean_roc_curve.txt'         using 3:4 with lines; \
	set output 'test_clean_best_precision_recall.png'; \
	set xlabel 'Recall'; \
	set ylabel 'Precision'; \
	plot [0:1][0:1] 'test_clean_pr_curve.txt'          using 3:4 with lines; \
"

eog test_clean_best_roc_curve.png &
eog test_clean_best_precision_recall.png &

set +x
