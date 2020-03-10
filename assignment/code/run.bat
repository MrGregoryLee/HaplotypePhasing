set example=%1
set size=%2

python unmask.py %example% %size% && Rscript calculate_switch_accuracy.R ../data/example_data_%example%_sol.txt ../output/example_data_%example%_block_%size%_sol.txt