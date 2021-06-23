@echo off 
setlocal enabledelayedexpansio
set local_path=C:\Users\OpenClassrooms\jupyter-notebooks\INGENIEUR_ML
set kernel[0]=michaelfumery/stackoverflow-questions-data-cleaning
set kernel[1]=michaelfumery/stackoverflow-questions-tag-generator

echo Local path : %local_path%

set "x=0"

:SymLoop
if defined kernel[%x%] (
    call echo Kernel Kaggle : %%kernel[%x%]%%
    call kaggle kernels pull %%kernel[%x%]%% -p %local_path%
    set /a "x+=1"
    GOTO :SymLoop
)


PAUSE
