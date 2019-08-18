FILE=$1

if [[ $FILE != "ntu_image_skeleton" && $FILE != "senz3d_image_skeleton" ]]; then
  echo "Available datasets are ntu_image_skeleton, senz3d_image_skeleton"
  exit 1
fi

echo "Specified [$FILE]"

URL=http://disi.unitn.it/~hao.tang/uploads/datasets/GestureGAN/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
