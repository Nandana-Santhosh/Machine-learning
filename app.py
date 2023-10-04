import tkinter as tk
from tkinter import filedialog, Text
import os

base = tk.Tk()
base.title("XBA")


canvas = tk.Canvas(base, width=400, height=300)
canvas.pack()

frame = tk.Frame(base, bg="black")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

base.attributes('-fullscreen', True)

models = []

if os.path.isfile('save.txt'):
    with open('save.txt', 'r') as f:
        tempModels = f.read()
        tempModels = tempModels.split(',')
        models = [x for x in tempModels if x.strip()]


# def addModel():
    
#     for widget in frame.winfo_children():
#         widget.destroy()

#     modelname=filedialog.askopenfilename(initialdir="/", title="select file",
#                                          filetypes = (("executables", "*py"), ("all files", "*.*")))

    
#     models.append(modelname)
#     print(modelname)
#     for model in models:
#         label = tk.Label(frame, text="Model", bg="white")
#         label.pack()



def classificationModel():
    
    os.system("classification.py")

# def trainmodel():

#     os.system("train.py")


    
    # models.append(modelname)
    # print(modelname)
    # for model in models:
    #     label = tk.Label(frame, text="Model", bg="white")
    #     label.pack()

    # for model in modelname:
    #     os.startfile(model)




# def runModel():
#     for model in models:
#         os.startfile(model)



openModel = tk.Button(base, text="Classification Model", padx=10,
                      pady=2, fg="white", activebackground="#21759b", font="arial", bg="#263D42", command=classificationModel)
openModel.pack()
openModel.place(x=680, y=375)


# runModel = tk.Button(base, text="Run Model", padx=19.7,
#                      pady=2, fg="white", activebackground="#21759b", font="arial", bg="#263D42", command=runModel)
# runModel.pack()
# runModel.place(x=700, y=425)


# trainmodel = tk.Button(base, text="Train Model", padx=10,
#                        pady=2, fg="white", activebackground="#21759b", font="arial", bg="#263D42", command=trainmodel)
# trainmodel.pack()

for model in models:
    label = tk.Label(frame, text=model)
    label.pack()


quitWindow = tk.Button(base, text="Quit", command=base.destroy, padx=20,
                       pady=2, fg="white", activebackground="#21759b", font="arial", bg="#263D42")
quitWindow.pack()
quitWindow.place(x=730, y=475)



base.mainloop()


with open('save.txt', 'w') as f:
    for model in models:
        f.write(model + ',')