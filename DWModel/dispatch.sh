for ii in {0..8} 
do
    cd ./iter.$(printf "%06d" $ii)
    for folder in *; do
        if [ -d "$folder" ]; then
            cd $folder
            # Will not run if no directories are available
            cp ~/data.gpfs/Github/ferro_scratch/PTO/DeepWannier/template/batch_submit.base ./
            ndata=$(find ./wannier/* -maxdepth 0 -type d | wc -l)
            ndata=$(($ndata - 1))
            sed "s/REPLACE0/$ndata/" batch_submit.base > batch_submit.slurm
            rm batch_submit.base
            cd ..
        fi
    done
    cd ..
done