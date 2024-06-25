# Once having the 'images' folder in root,
# run this shell program to seperate Apples and Grapes leaves

rm -rf Grapes
mkdir Grapes
cp -r images/Grape_Black_rot Grapes/Grape_Black_rot
cp -r images/Grape_Esca Grapes/Grape_Esca
cp -r images/Grape_healthy Grapes/Grape_healthy
cp -r images/Grape_spot Grapes/Grape_spots

rm -rf Apples
mkdir Apples
cp -r images/Apple_Black_rot Apples/Apple_Black_rot
cp -r images/Apple_healthy Apples/Apple_healthy
cp -r images/Apple_rust Apples/Apple_rust
cp -r images/Apple_scab Apples/Apple_scab