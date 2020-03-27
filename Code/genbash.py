pythonFileName = "Merge_Sequential_Classification.py"
algs = ["srnc", "svm", "dt", "lr"]
choices = ["0", "1", "mean"]

fi = open("runme.sh", "w")
for choice in choices:
    for alg in algs:
        for seed_number in range(10):
            fi.write("python " + pythonFileName + " " + str(seed_number) + " " + choice+ " "+alg+" &\n")
fi.write("echo done!\n")
fi.close()
