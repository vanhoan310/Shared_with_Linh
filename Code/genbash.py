pythonFileName = "Merge_Sequential_Classification.py"
# pythonFileName = "ARI_Sequential_Classification.py"
# algs = ["srnc", "svm", "dt", "lr", "LinearSVM"]
# choices = ["0", "1", "mean"]
algs = ["LinearSVM"]
choices = ["0"]

fi = open("runme2.sh", "w")
for choice in choices:
    for alg in algs:
        # for dataId in range(6):
        for dataId in [4,5]:
            for seed_number in range(10):
                fi.write("python " + pythonFileName + " " + str(seed_number)+ " "+str(dataId)+ " &\n")
                # fi.write("python " + pythonFileName + " " + str(seed_number) + " " + choice+ " "+alg+"\n")
                # fi.write("python " + pythonFileName + " " + str(seed_number) + " " + choice+ " "+alg+" &\n")
    fi.write("echo done!\n")
    fi.write("echo done!\n")
fi.close()
