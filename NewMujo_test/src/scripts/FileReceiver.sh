exp_id=14
flag=$1
case ${flag} in
    "ETG")
        mode="ETG"
        ;;
    *)
        mode="ETG_RL"
        ;;
esac
scp -r niujh@222.200.180.49:/home/niujh/ForCode/NewMujo_test/src/data/exp${exp_id}_${mode}_models \
    /home/niujh/ForCode/NewMujo_test/src/data