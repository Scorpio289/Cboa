import pymysql as pm
import re
import numpy
from sklearn.cluster import KMeans
from pylab import figure, subplot, hist, xlim, show,plot
at = [[0 for i in range(70)] for i in range(15040)]
#连接数据库
cnx = pm.connect(user='root',password='260877289',host='127.0.0.1',database='moviedata')
#演员数量下标
acotrnum=0
#数据库语句
cur1 = cnx.cursor()
cur1.execute("select * from movie")

cur2 = cnx.cursor()
cur2.execute("select ActorID from actor")
#while cur.fetchone() is not None :
data=cur1.fetchone()
#print(len(cur2.fetchall()))
actorinfo=cur2.fetchone()
#ai=list(actorinfo)

print(actorinfo)
for ij in range(len(at)):
    actorinfo = cur2.fetchone()
    if actorinfo is not None:
        at[ij][0]=int(re.sub(repl="\D",string= "",count=list(actorinfo)[0]))#正则
print(at[0][0])
print(at[10][0])
print(at[10][10])

while data is not None :
    
    
    
    t=[0 for i in range(70)]
    data=cur1.fetchone()
    if data is not None:
        #print(data)
        movieID=data[0]
        movietype=data[3]
        #print(movietype)
        if '爱情' in movietype :
            t[1]=1
        if '歌舞' in movietype :
            t[2]=1
        if '剧情' in movietype :
            t[3]=1
        if '纪录片' in movietype:
            t[4] = 1
        if '传记' in movietype:
            t[5] = 1
        if '动作' in movietype:
            t[6] = 1
        if '冒险' in movietype:
            t[7] = 1
        if '犯罪' in movietype:
            t[8] = 1
        if '青春' in movietype:
            t[9] = 1
        if '武侠' in movietype:
            t[10] = 1
        if '古装' in movietype:
            t[11] = 1
        if '奇幻' in movietype:
            t[12] = 1
        if '动画' in movietype:
            t[13] = 1
        if '科幻' in movietype:
            t[14] = 1
        if '惊悚' in movietype:
            t[15] = 1
        if '战争' in movietype:
            t[16] = 1
        if '悬疑' in movietype:
            t[17] = 1
        if '喜剧' in movietype:
            t[18] = 1
        if '运动' in movietype:
            t[19] = 1
        if '亲情' in movietype:
            t[20] = 1
        if '穿越' in movietype:
            t[21] = 1
        if '灾难' in movietype:
            t[22] = 1
        if '侦探' in movietype:
            t[23] = 1
        if '神秘' in movietype:
            t[24] = 1
        if '动物' in movietype:
            t[25] = 1
        if '恐怖' in movietype:
            t[26] = 1
        if '功夫' in movietype:
            t[27] = 1
        if '励志' in movietype:
            t[28] = 1
        if '公路' in movietype:
            t[29] = 1
        if '西部' in movietype:
            t[30] = 1
        if '都市' in movietype:
            t[31] = 1
        if '时尚' in movietype:
            t[32] = 1
        if '职场' in movietype:
            t[33] = 1
        if '警匪' in movietype:
            t[34] = 1
        if '舞台艺术片' in movietype:
            t[35] = 1
        if '儿童' in movietype:
            t[36] = 1
        if '军事' in movietype:
            t[37] = 1
        if '音乐' in movietype:
            t[38] = 1
        if '怀旧' in movietype:
            t[39] = 1
        if '玄幻' in movietype:
            t[40] = 1
        if '革命' in movietype:
            t[41] = 1
        if '军旅' in movietype:
            t[42] = 1
        if '友情' in movietype:
            t[43] = 1
        if '家庭' in movietype:
            t[44] = 1
        if '历史' in movietype:
            t[45] = 1
        if '农村' in movietype:
            t[46] = 1
        if '商战' in movietype:
            t[47] = 1
        if '同性' in movietype:
            t[48] = 1
        if '涉案' in movietype:
            t[49] = 1
        if '贺岁' in movietype:
            t[50] = 1
        if '抢险' in movietype:
            t[51] = 1
        if '灵异' in movietype:
            t[52] = 1
        if '主旋律' in movietype:
            t[53] = 1
        if '女性' in movietype:
            t[54] = 1
        if '文艺' in movietype:
            t[55] = 1
        if '黑色' in movietype:
            t[56] = 1
        if '心理' in movietype:
            t[57] = 1
        if '戏曲' in movietype:
            t[58] = 1
        if '民族' in movietype:
            t[59] = 1
        if '反腐' in movietype:
            t[60] = 1
        if '真人秀' in movietype:
            t[61] = 1
        if '综艺' in movietype:
            t[62] = 1
        if '大电影' in movietype:
            t[63] = 1
        if '枪战' in movietype:
            t[64] = 1
        if '黑帮' in movietype:
            t[65] = 1
        if '科教片' in movietype:
            t[66] = 1
        if '神话' in movietype:
            t[67] = 1
        if '幽默' in movietype:
            t[68] = 1
        if '竞技' in movietype:
            t[69] = 1
        cur3 = cnx.cursor()
        cur3.execute("select ActorID from movie_actor where MovieID ='"+movieID+"'")
        actorid=0
        sum=0
        while actorid is not None:
            actorid=cur3.fetchone()
            if actorid is not None:
                actornum=int(re.sub("\D", "",list(actorid)[0]))#正则
                for i in range(len(at)):
                    if at[i][0]==actornum:
                        for j in range(len(t)):
                            at[i][j]+=t[j]
                        break
                #print(at[0])

cnx.close()
#at.savetxt("actor_type.txt",a)

"""
kmeans = KMeans(3, init='random') # initialization
kmeans.fit(at) # actual execution
c=KMeans.predict()

figure()
plot(data[c==1,0],data[c==1,2],'bo',alpha=.7)
plot(data[c==2,0],data[c==2,2],'go',alpha=.7)
plot(data[c==0,0],data[c==0,2],'mo',alpha=.7)
show()

"""
import pandas as pd
data1 = pd.DataFrame(at)
data1.to_csv('data_notype.csv',na_rep='NA',header=0,index=0)
print("ok")