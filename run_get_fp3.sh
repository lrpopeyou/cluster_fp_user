HADOOP_HOME="/home/map/hw/hadoop-nl/hadoop"
OUTPUT_NL=/app/lbs/lbs-da-off/traj/


PYTAR="/app/lbs/lbs-guiji/ident_traj/tools/py.tar#py27"

OUTPUT_DETAIL=stay_ap.detail
INPUT=stay_ap.beijing.byuser
OUTPUT=stay_ap.beijing2
$HADOOP_HOME/bin/hadoop dfs -conf ~/client/hadoop-site-ares.xml  -rmr $OUTPUT_NL/$OUTPUT_DETAIL
$HADOOP_HOME/bin/hadoop dfs -conf ~/client/hadoop-site-ares.xml  -rmr $OUTPUT_NL/$OUTPUT

#                -input /app/lbs/lbs-guiji/ident_traj/mark_i2c/2014030*  \
#                -input /app/lbs/lbs-guiji/ident_traj/mark_i2c/201402*  \
#                -input /app/lbs/lbs-guiji/ident_traj/mark_i2c/201401*  \
#                -input /app/lbs/lbs-guiji/ident_traj/mark_i2c/20140301  \
#-jobconf stream.reduce.streamprocessor.2="sh -x py27/py27.sh cluster_by_grid.py" \

#                -jobconf abaci.is.dag.job=true \
#                -jobconf abaci.dag.vertex.num=3 \
#                -jobconf abaci.dag.next.vertex.list.0=1 \
#                -jobconf abaci.dag.next.vertex.list.1=2 \
 
$HADOOP_HOME/bin/hadoop streaming \
                -input $OUTPUT_NL/$INPUT \
               -output $OUTPUT_NL/$OUTPUT \
                -mapper "sh -x py27/py27.sh cluster_by_user.py" \
                -reducer "sh -x py27/py27.sh cluster_by_grid.py" \
                -jobconf ares.user=zhuozhengxing \
               -jobconf hadoop.job.ugi=lbs-da-off,lbs-da-off \
                -jobconf mapred.job.name="get_fp_from_3mnonth_trajectory_fromuser" \
                -jobconf mapred.reduce.tasks=8999 \
                -jobconf mapred.job.map.capacity=2000 \
                -jobconf mapred.min.split.size=1000000000 \
                -jobconf mapred.job.reduce.capacity=1999\
                -jobconf stream.memory.limit=2000 \
                -file sh/get_ap_list.py \
                -file sh/cluster_by_grid.py \
                -file sh/grid2.py \
                -file sh/test_pic.py \
                -file sh/optics.py \
                -file sh/util.so \
                -file sh/count_users_var.py \
                -file sh/grid.py \
                -file sh/get_point_stay.py \
                -file sh/common.py \
                -file sh/cluster_by_user.py \
                -cacheArchive  $PYTAR 
#
#-jobconf mapred.output.compress=true \
#-partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner -jobconf stream.num.map.output.key.fields=2 -jobconf num.key.fields.for.partition=1  \
#-reducer "sh -x py27/py27.sh cluster_by_user.py byuser |sh -x py27/py27.sh cluster_by_grid.py " \
#-input /app/lbs/lbs-guiji/ident_traj/mark/201401*/  \
#-reducer "uniq|sh -x py27/py27.sh count_users_var.py"  \
#-mapper "grep STAY|sh -x py27/py27.sh get_ap_list.py|uniq" \
#-input /app/lbs/lbs-guiji/ident_traj/mark_i2c/20131* \
#-D stream.reduce.streamprocessor.1="sh -x py27/py27.sh get_ap_list.py" \

#-input /app/lbs/lbs-guiji/ident_traj/mark_i2c/20131*/  \

#                -D abaci.is.dag.job=true \
#                -D abaci.dag.vertex.num=2 \
#                -D abaci.dag.next.vertex.list.0=1 \
#                -D abaci.dag.next.vertex.list.1=2 \
#-D mapred.output.dir.0=$OUTPUT_NL/$OUTPUT_DETAIL \
 
#                -input /app/lbs/lbs-guiji/ident_traj/mark/201401*/  \
#-D abaci.dag.autosize.reduce.tasks=true \
                


