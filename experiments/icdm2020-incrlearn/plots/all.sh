for i in nomao phishing wall_robot mnist fashion news ldpa cifar10 cifar100; do
    echo $i
    cd $i
    python ../plots.py
    cd ..
done
