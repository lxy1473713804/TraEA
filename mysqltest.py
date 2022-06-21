import pymysql
#打开数据库连接
conn = pymysql.connect(host='localhost',user = "root",passwd = "gis6391700",db = "geotradb")
#获取游标
cursor=conn.cursor()
sql = "INSERT INTO geopiccoor (objid, type, piccoor,time,piclocal) VALUES ( '%s', '%s', '%s', '%s', '%s' )"
data = ('52', 'motorcycle', 'akckaoncokancokanfcishdfia565+64646565656565455656+565656565623464656565656565','2021-11-18 16:46:53')
cursor.execute(sql % data)
conn.commit()
print('成功插入', cursor.rowcount, '条数据')
