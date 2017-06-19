import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix

/**
  * Created by shengmingqijiquan on 2017/6/19 0019.
  */
object ModelGenerator {
  def main(args: Array[String]) {
    /** 第一步：初始化spark配置
    * 1.新建常量sparkConf
    * 2.根据sparkConf配置信息生成sparkContext
    * */
    println("------------------第一步：初始化spark配置-------------")
    //初始化配置
    val sparkConf = new SparkConf()
      .setMaster("spark://Master:7077")
      .setAppName("ModelGenerator")
      .set("spark.akka.frameSize", "2000")
      .set("spark.network.timeout", "1200")
    val sparkContext = new SparkContext(sparkConf)

    /** 第二步：导入数据集
      * 1.导入数据集，并筛选出所需要的的数据集
      * */
    println("------------------第二步：导入本地数据集----------------")
    val rawData = sparkContext.textFile("G:\\ml-100k\\u.data")
    rawData.first()//返回第一个元素
    /* 取出u.data中除了时间戳的数据*/
    val rawRatings = rawData.map(_.split("\t").take(3))
    /*使用scala模式匹配的将原始的评级数据转换为所需要的RDD[Rating]*/
    val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }

    /** 第三步：训练模型
      * 1.根据筛选的数据集，训练推荐模型
      * 2.设置ALS的参数
      *     rank=50, iterations=10, lambda=0.01
      *     rank：对应ALS模型中的因子个数，也就是在低阶近似矩阵中的隐含特征个
      *     iterations：对应运行时的迭代次数
      *     lambda：该参数控制模型的正则化过程，从而控制模型的过拟合情况。其值越高，正则化越严厉
      * */
    println("------------------第三步：训练模型----------------")
    /*ALS.train()返回一个MatrixFactorizationModel对象，该对象将用户因子和物品因子分别保存在一个（id，factor）对类型的RDD中。
    * 它们分别称作userFeatures和productFeature*/
    val model = ALS.train(ratings,50,10,0.01);
    /* Inspect the user factors */
    model.userFeatures;
    /* Count user factors and force computation */
    model.userFeatures.count();
    model.productFeatures.count();
    /* Make a prediction for a single user and movie pair */
    val predictRating = model.predict(789,123);
    /* Make predictions for a single user across all movies */
    val userId = 789
    val K = 10
    val topKRecs = model.recommendProducts(userId,K);//为了789推荐前10产品
    println(topKRecs.mkString("\n"));//转成字符串
    /**
      * 第四部：验证模型
      * 进行推荐内容的检验，要直观地检验推荐的效果，可以简单比对下用户所评级过的电影的标题和被推荐的那些电影的电影
      */
    println("---------------------第四步：验证模型=====================================");
    val movies = sparkContext.textFile("G:\\ml-100k\\u.item");
    val titles = movies.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt,array(1))).collectAsMap();
    val moviesForUser = ratings.keyBy(_.user).lookup(789);//在上面的评级中找789用户
    println(moviesForUser.size)//该用户对多少电影评级过
    moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product),rating.rating)).foreach(println);//实际中789用户已经看的
    topKRecs.map(rating => (titles(rating.product),rating.rating)).foreach(println);//推荐系统中推荐的 通过上下对比
    println("===========================推荐检验结束================================================");
    /**
      * 第五步：使用模型进行推荐
      * 进行物品推荐 ，利用余弦相似度来对指定物品的因子向量与其他物品的做比
      * */
    println("---------------------第五步：使用模型进行推荐=====================================");
    val itemId = 567;
    val itemFactor = model.productFeatures.lookup(itemId).head;
    val itemVector = new DoubleMatrix(itemFactor);

    //现在求各个物品的余弦相似度：
    val sims = model.productFeatures.map{case (id,factor) =>
      val factorVector = new DoubleMatrix(factor);
      val sim = cosineSimilarit(itemVector,factorVector);
      (id,sim);//返回
    }
    val sortedSims = sims.top(K)(Ordering.by[(Int, Double), Double]{case (id,similiarity) => similiarity});//通过相似度
    println(sortedSims.mkString("\n"));

    //检查推荐的相似物品
    println(titles(itemId));
    //这一次我们取前11部最相似电影，以排除给定的那部。所以，可以选取列表中的第1到11项
    val sortedSims2 = sims.top(K+1)(Ordering.by[(Int, Double), Double]{case (id,similiarity) => similiarity});
    println(sortedSims.slice(1,11).map{case (id,sim) => (titles(id),sim)}.mkString("\n"));
    println("=======================物品推荐完成====================================================")

    /**
      * 第六步：模型评估
      * 推荐模型效果的评估
      * Compute squared error between a predicted and actual rating ，We'll take the first rating for our example user 789
      * */
    println("================================第六步：模型评估=========================================")

    val actualRating = moviesForUser.take(1)(0)
    println(actualRating.rating);
    //然后，求模型的预计评级
    val predictedRating = model.predict(789,actualRating.product);
    println(predictedRating);
    //最后，我们计算实际评级和预计评级的平方误差
    val squaredError = math.pow(actualRating.rating-predictedRating,2.0);
    //整个数据集上的MSE，需要对每一条(user, movie, actual rating, predictedrating)记录都计算该平均误差，然后求和，再除以总的评级次数
    val usersProducts = ratings.map{case Rating(user, product, rating)  => (user, product)};
    val predictions = model.predict(usersProducts).map{case Rating(user, product, rating) => ((user, product), rating)}
    //这个RDD的主键为“用户物品”对，键值为相应的实际评级和预计评级。
    val ratingsAndPredictions = ratings.map{ case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions)
    //最后，求上述MSE
    var MSE = ratingsAndPredictions.map{case((user,product),(actual,predicted)) => math.pow(actual-predicted,2.0)}.reduce(_+_)/ratingsAndPredictions.count();
    println("Mean Squared Error = "+MSE);
    //均跟方误差
    val RMSE = math.sqrt(MSE)
    println("Root Mean Squared Error = " + RMSE)
    println("=====================s使用MLlib下的RegressionMetrics和RankingMetrics+======================")
    //实际中MLlib下的RegressionMetrics和RankingMetrics类也提供了相应的函数
    //RegressionMetrics来求解MSE和RMSE得分
    val predictedAndTrue = ratingsAndPredictions.map{case((user,product),(actual,predicted)) => (actual,predicted)};
    val regressionMetrics = new RegressionMetrics(predictedAndTrue);
    //之后就可以查看各种指标的情况，包括MSE和RMSE。
    println(regressionMetrics.meanSquaredError);
    println(regressionMetrics.rootMeanSquaredError);
  }
  //求两个向量的余弦度，。1表示完全相似，0表示两者互不相关（即无相似性）
  def cosineSimilarit(vec1: DoubleMatrix, vec2: DoubleMatrix):Double={
    vec1.dot(vec2)/(vec1.norm2() * vec2.norm2());
  }
}
