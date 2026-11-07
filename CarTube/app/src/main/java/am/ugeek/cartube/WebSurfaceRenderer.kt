package am.ugeek.cartube

import android.app.Presentation
import android.content.Context
import android.content.SharedPreferences
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.os.SystemClock
import android.view.InputDevice
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import androidx.car.app.CarContext
import androidx.car.app.SurfaceCallback
import androidx.car.app.SurfaceContainer

/**
 * Draws a WebView onto the car's surface.
 *
 * The car hands us a Surface; we back a private VirtualDisplay with it and show a
 * Presentation (a window on that display) containing the WebView. Touches the car
 * reports on the surface are replayed into the window as MotionEvents.
 */
class WebSurfaceRenderer(private val carContext: CarContext) : SurfaceCallback {

    private val prefs: SharedPreferences =
        carContext.getSharedPreferences(Prefs.NAME, Context.MODE_PRIVATE)

    private var virtualDisplay: VirtualDisplay? = null
    private var presentation: Presentation? = null
    private var root: FrameLayout? = null
    private var webView: WebView? = null

    private val prefListener = SharedPreferences.OnSharedPreferenceChangeListener { p, key ->
        if (key == Prefs.KEY_PENDING_NONCE) {
            p.getString(Prefs.KEY_PENDING_URL, null)?.let { load(it) }
        }
    }

    init {
        prefs.registerOnSharedPreferenceChangeListener(prefListener)
    }

    // ---- surface lifecycle -------------------------------------------------

    override fun onSurfaceAvailable(container: SurfaceContainer) {
        val surface = container.surface ?: return
        if (container.width <= 0 || container.height <= 0) return
        releaseDisplay()

        val displayManager = carContext.getSystemService(DisplayManager::class.java)
        val display = displayManager.createVirtualDisplay(
            "CarTube", container.width, container.height, container.dpi, surface, 0
        )
        virtualDisplay = display

        val p = Presentation(carContext, display.display, android.R.style.Theme_DeviceDefault_NoActionBar)
        val frame = FrameLayout(p.context)
        val web = createWebView(p.context, frame)
        frame.addView(web, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        p.setContentView(frame)
        p.show()

        presentation = p
        root = frame
        webView = web
        web.loadUrl(prefs.getString(Prefs.KEY_LAST_URL, null) ?: Prefs.HOME)
    }

    override fun onSurfaceDestroyed(container: SurfaceContainer) {
        releaseDisplay()
    }

    fun release() {
        prefs.unregisterOnSharedPreferenceChangeListener(prefListener)
        releaseDisplay()
    }

    private fun releaseDisplay() {
        webView?.let { web ->
            (web.parent as? FrameLayout)?.removeView(web)
            web.destroy()
        }
        webView = null
        root = null
        presentation?.dismiss()
        presentation = null
        virtualDisplay?.release()
        virtualDisplay = null
    }

    private fun createWebView(context: Context, frame: FrameLayout): WebView =
        WebView(context).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.mediaPlaybackRequiresUserGesture = false
            settings.useWideViewPort = true
            settings.loadWithOverviewMode = true

            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView, url: String) {
                    prefs.edit().putString(Prefs.KEY_LAST_URL, url).apply()
                }
            }

            // Full-screen video: YouTube hands us a separate view to show on top.
            webChromeClient = object : WebChromeClient() {
                private var fullscreen: View? = null

                override fun onShowCustomView(view: View, callback: CustomViewCallback) {
                    fullscreen?.let { frame.removeView(it) }
                    fullscreen = view
                    frame.addView(view, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
                }

                override fun onHideCustomView() {
                    fullscreen?.let { frame.removeView(it) }
                    fullscreen = null
                }
            }
        }

    // ---- navigation --------------------------------------------------------

    fun load(url: String) {
        prefs.edit().putString(Prefs.KEY_LAST_URL, url).apply()
        webView?.loadUrl(url)
    }

    fun goBack() {
        webView?.let { if (it.canGoBack()) it.goBack() }
    }

    fun reload() {
        webView?.reload()
    }

    // ---- input from the car ------------------------------------------------

    override fun onClick(x: Float, y: Float) {
        val target = root ?: return
        val downTime = SystemClock.uptimeMillis()
        dispatch(target, downTime, downTime, MotionEvent.ACTION_DOWN, x, y)
        dispatch(target, downTime, downTime + 60, MotionEvent.ACTION_UP, x, y)
    }

    override fun onScroll(distanceX: Float, distanceY: Float) {
        webView?.evaluateJavascript("window.scrollBy($distanceX, $distanceY);", null)
    }

    override fun onFling(velocityX: Float, velocityY: Float) {
        val dx = -velocityX / 4
        val dy = -velocityY / 4
        webView?.evaluateJavascript(
            "window.scrollBy({left: $dx, top: $dy, behavior: 'smooth'});", null
        )
    }

    override fun onScale(focusX: Float, focusY: Float, scaleFactor: Float) {
        if (scaleFactor > 0.01f && scaleFactor < 100f) webView?.zoomBy(scaleFactor)
    }

    private fun dispatch(target: View, downTime: Long, time: Long, action: Int, x: Float, y: Float) {
        val event = MotionEvent.obtain(downTime, time, action, x, y, 0)
        event.source = InputDevice.SOURCE_TOUCHSCREEN
        target.dispatchTouchEvent(event)
        event.recycle()
    }
}
