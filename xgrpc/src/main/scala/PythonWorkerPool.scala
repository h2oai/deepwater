package deepwater.utils

import java.io.File
import java.net.Socket

import scala.collection.mutable
import scala.collection.JavaConversions.mapAsScalaMap

import java.util.HashMap;

class PythonWorkerPool() extends Logging {

  var isStopped = false
  private val pythonWorkers = mutable.HashMap[(String, Map[String, String]), PythonWorkerFactory]()

  var driverTmpDir: Option[String] = None

  def stop() {

    if (!isStopped) {
      isStopped = true
      pythonWorkers.values.foreach(_.stop())

      // If we only stop sc, but the driver process still run as a services then we need to delete
      // the tmp dir, if not, it will create too many tmp dirs.
      // We only need to delete the tmp dir create by driver
      driverTmpDir match {
        case Some(path) =>
          try {
            Utils.deleteRecursively(new File(path))
          } catch {
            case e: Exception =>
              logWarning(s"Exception while deleting Spark temp dir: $path", e)
          }
        case None => // We just need to delete tmp dir created by driver, so do nothing on executor
      }
    }
  }


  def createPythonWorker(pythonExec: String, envVars: HashMap[String, String]): java.net.Socket = {
    synchronized {
      val env:Map[String, String]  = mapAsScalaMap(envVars).toMap
      val key = (pythonExec, env)
      pythonWorkers.getOrElseUpdate(key, new PythonWorkerFactory(pythonExec, env)).create()
    }
  }


  def destroyPythonWorker(pythonExec: String, envVars: HashMap[String, String], worker: Socket) {
    synchronized {
      val env:Map[String, String]  = mapAsScalaMap(envVars).toMap
      val key = (pythonExec, env)
      pythonWorkers.get(key).foreach(_.stopWorker(worker))
    }
  }


  def releasePythonWorker(pythonExec: String, envVars: HashMap[String, String], worker: Socket) {
    synchronized {
      val env:Map[String, String]  = mapAsScalaMap(envVars).toMap
      val key = (pythonExec, env)
      pythonWorkers.get(key).foreach(_.releaseWorker(worker))
    }
  }
}