for dir in extra mesh pressure velocity;
do
  cd $dir
  rm ../../assets/$dir/*.webp
  for file in *.png;
  do
    convert $file -scale 30% -quality 50 "`basename $file .png`.webp"; 
  done
  mv *.webp ../../assets/$dir/
  cd ..
done
