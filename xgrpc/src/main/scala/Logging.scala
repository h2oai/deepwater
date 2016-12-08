package deepwater.utils

trait Logging {

  // Make the log field transient so that objects with Logging can
  // be serialized and used on another machine

  // Method to get the logger name for this object
  protected def logName = {
    // Ignore trailing $'s in the class names for Scala objects
    this.getClass.getName.stripSuffix("$")
  }

  // Method to get or create the logger for this object

  // Log methods that take only a String
  protected def logInfo(msg: => String)
  {
  }

  protected def logDebug(msg: => String)
  {
  }

  protected def logTrace(msg: => String)
  {
  }

  protected def logWarning(msg: => String)
  {
  }

  protected def logError(msg: => String)
  {
  }

  // Log methods that take Throwables (Exceptions/Errors) too
  protected def logInfo(msg: => String, throwable: Throwable)
  {
  }

  protected def logDebug(msg: => String, throwable: Throwable)
  {
  }

  protected def logTrace(msg: => String, throwable: Throwable)
  {
  }

  protected def logWarning(msg: => String, throwable: Throwable)
  {
  }

  protected def logError(msg: => String, throwable: Throwable)
  {
  }

  protected def isTraceEnabled(): Boolean = {
    true
  }

  protected def initializeLogIfNecessary(isInterpreter: Boolean): Unit = {
    if (!Logging.initialized) {
    }
  }

}

private object Logging {
  @volatile private var initialized = false
}