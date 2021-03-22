for x in *.comp; do
    glslc -O $x -o $x.spv &
done
wait
