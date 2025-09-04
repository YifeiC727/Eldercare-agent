// MongoDB初始化脚本
// 创建数据库和用户

// 切换到eldercare数据库
db = db.getSiblingDB('eldercare');

// 创建应用用户
db.createUser({
  user: 'eldercare_user',
  pwd: 'eldercare_pass',
  roles: [
    {
      role: 'readWrite',
      db: 'eldercare'
    }
  ]
});

// 创建集合和索引
db.createCollection('users');
db.createCollection('conversations');
db.createCollection('questions');
db.createCollection('emotion_trends');
db.createCollection('keyword_memory');

// 为用户集合创建索引
db.users.createIndex({ "login_info.username": 1 }, { unique: true });
db.users.createIndex({ "created_at": 1 });
db.users.createIndex({ "last_login": 1 });

// 为对话集合创建索引
db.conversations.createIndex({ "user_id": 1 });
db.conversations.createIndex({ "created_at": 1 });
db.conversations.createIndex({ "conversation_id": 1 });

// 为情感趋势创建索引
db.emotion_trends.createIndex({ "user_id": 1, "timestamp": 1 });
db.emotion_trends.createIndex({ "timestamp": 1 });

// 为关键词记忆创建索引
db.keyword_memory.createIndex({ "user_id": 1, "keyword": 1 });
db.keyword_memory.createIndex({ "created_at": 1 });

print('✅ MongoDB数据库初始化完成');
print('📊 数据库: eldercare');
print('👤 应用用户: eldercare_user');
print('🔑 密码: eldercare_pass');


