export OMP_NESTED=TRUE
export OMP_MAX_ACTIVE_LEVELS=4
while [[ $# -gt 1 ]]
do
key="$1"
case $key in
    --exe)
    EXEC="$2"
    shift
    ;;
    --prefix)
    PREFIX="$2"
    shift
    ;;
    --log-path)
    LOG_PATH="$2"
    shift # past argument
    ;;
    --cfg)
    EXP_CFG="$2"
    shift # past argument
    ;;
    --pan)
    PAN_ID="$2"
    shift # past argument
    ;;
    --thread)
    THREAD_NUM="$2"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
shift # past argument or value
done
module load intel
${EXEC} --cfg ${EXP_CFG} --pan ${PAN_ID} --thread ${THREAD_NUM} > ${LOG_PATH}/${PREFIX}_${PAN_ID}.log