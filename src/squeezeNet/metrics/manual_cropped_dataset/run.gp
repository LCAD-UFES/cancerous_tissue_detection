set terminal png size 800,600; 
set output 'val_roc_curve.png';                     set xlabel '1 - Specificity'; set ylabel 'Sensitivity'; plot [0:1][0:1] 'val_metrics.txt'            using 10:11
set output 'val_precision_recall.png';              set xlabel 'Recall';          set ylabel 'Precision';   plot [0:1][0:1] 'val_metrics.txt'            using 8:7 
set output 'test_roc_curve.png';                    set xlabel '1 - Specificity'; set ylabel 'Sensitivity'; plot [0:1][0:1] 'test_metrics.txt'           using 10:11
set output 'test_precision_recall.png';             set xlabel 'Recall';          set ylabel 'Precision';   plot [0:1][0:1] 'test_metrics.txt'           using 8:7 
set output 'val_best_roc_curve.png';                set xlabel '1 - Specificity'; set ylabel 'Sensitivity'; plot [0:1][0:1] 'val_roc_curve.txt'          using 3:4 with lines;
set output 'val_best_precision_recall.png';         set xlabel 'Recall';          set ylabel 'Precision';   plot [0:1][0:1] 'val_pr_curve.txt'           using 3:4 with lines; 
set output 'test_best_roc_curve.png';               set xlabel '1 - Specificity'; set ylabel 'Sensitivity'; plot [0:1][0:1] 'test_roc_curve.txt'         using 3:4 with lines;
set output 'test_best_precision_recall.png';        set xlabel 'Recall';          set ylabel 'Precision';   plot [0:1][0:1] 'test_pr_curve.txt'          using 3:4 with lines; 
set output 'val_best_roc_curve_0.0025.png';         set xlabel '1 - Specificity'; set ylabel 'Sensitivity'; plot [0:1][0:1] 'val_roc_curve_0.0025.txt'   using 3:4 with lines; 
set output 'val_best_precision_recall_0.0025.png';  set xlabel 'Recall';          set ylabel 'Precision';   plot [0:1][0:1] 'val_pr_curve_0.0025.txt'    using 3:4 with lines; 
set output 'test_best_roc_curve_0.0025.png';        set xlabel '1 - Specificity'; set ylabel 'Sensitivity'; plot [0:1][0:1] 'test_roc_curve_0.0025.txt'  using 3:4 with lines; 
set output 'test_best_precision_recall_0.0025.png'; set xlabel 'Recall';          set ylabel 'Precision';   plot [0:1][0:1] 'test_pr_curve_0.0025.txt'   using 3:4 with lines; 
set output 'val_best_roc_curve_001.png';            set xlabel '1 - Specificity'; set ylabel 'Sensitivity'; plot [0:1][0:1] 'val_roc_curve_0.001.txt'    using 3:4 with lines; 
set output 'val_best_precision_recall_001.png';     set xlabel 'Recall';          set ylabel 'Precision';   plot [0:1][0:1] 'val_pr_curve_0.001.txt'     using 3:4 with lines; 
set output 'test_best_roc_curve_001.png';           set xlabel '1 - Specificity'; set ylabel 'Sensitivity'; plot [0:1][0:1] 'test_roc_curve_0.001.txt'   using 3:4 with lines; 
set output 'test_best_precision_recall_001.png';    set xlabel 'Recall';          set ylabel 'Precision';   plot [0:1][0:1] 'test_pr_curve_0.001.txt'    using 3:4 with lines;

set output 'train_best_roc_curve.png';              set xlabel '1 - Specificity'; set ylabel 'Sensitivity'; plot [0:1][0:1] 'train_roc_curve.txt'        using 3:4 with lines;
set output 'train_best_precision_recall.png';       set xlabel 'Recall';          set ylabel 'Precision';   plot [0:1][0:1] 'train_pr_curve.txt'         using 3:4 with lines; 
 
