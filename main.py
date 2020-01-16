import fitting,modified_WA

if __name__ == "__main__":
    # initial params (you dont have to input)
    orientations = ["110","200","211","220","310","222"]

    # initial params (you have to input)
    # file_name 
    file_name = "peaks.csv"
    # ka lamda
    ka = [0.70926, 0.71354]
    # fitting functon {"gauss","lorentz","voigt"}
    function = "lorentz"
    output_dir = "./data/fitting/output/predict/"+function
    for orientation in orientations:
        # check fitting 
        ins = fitting.Main(path="./data/fitting/input/"+file_name,
                           file_name=file_name,
                           orientation = orientation, kalpha=ka, function=function,
                           output_dir = output_dir)
        # to csv
        ins.output()

    predict_dir = "./data/fitting/output/predict/lorentz"
    output_dir = "./data/modified_WA/output/graphs"
    # ["110","200","211","220","310","222"]
    orientations = ["110","200","211","220","310","222"]
    remove_orientations = ["110"]
    
    ins = modified_WA.Main(predict_dir = predict_dir,
               output_dir = output_dir,
               orientations = orientations,
               remove_orientations = remove_orientations,
               ka_lambda = 0.070931,
               C_h00 = 0.285)
