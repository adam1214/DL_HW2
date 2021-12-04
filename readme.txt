1. 安裝套件: 安裝numpy, torch, sklearn, PIL與matplotlib套件
2. 執行程式: 於資料夾根目錄執行main.py即可，無須輸入引數
3. 執行結果: 產出gen_data.npy, gen_label.npy與Train_Loss.png皆在資料夾根目錄。於./sample_image資料夾底下可分別看到9種類別(./class_0, ./class_1, ...)透過AutoEncoder與Gaussian noise產出的5種結果，分別是Gaussian_1.png, Gaussian_2.png, ... , Gaussian_5.png，而原先被sample出來當model input的圖命名為original.png