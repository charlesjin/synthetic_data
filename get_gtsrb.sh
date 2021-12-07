#!/bin/bash

mkdir temp
cd temp
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
unzip GTSRB_Final_Test_Images.zip
cd ..
mv temp/GTSRB/Final_Test/Images gtsrb/data/GTSRB_Test/Images
rm -rf temp


