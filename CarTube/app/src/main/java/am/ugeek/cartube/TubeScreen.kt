package am.ugeek.cartube

import androidx.car.app.AppManager
import androidx.car.app.CarContext
import androidx.car.app.Screen
import androidx.car.app.model.Action
import androidx.car.app.model.ActionStrip
import androidx.car.app.model.OnClickListener
import androidx.car.app.model.Template
import androidx.car.app.navigation.model.NavigationTemplate
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner

/** The car screen: a full-screen surface with the web page, plus a small button strip. */
class TubeScreen(carContext: CarContext) : Screen(carContext) {

    private val renderer = WebSurfaceRenderer(carContext)

    init {
        lifecycle.addObserver(object : DefaultLifecycleObserver {
            override fun onCreate(owner: LifecycleOwner) {
                carContext.getCarService(AppManager::class.java).setSurfaceCallback(renderer)
            }

            override fun onDestroy(owner: LifecycleOwner) {
                renderer.release()
            }
        })
    }

    override fun onGetTemplate(): Template {
        val actions = ActionStrip.Builder()
            .addAction(textAction("Back") { renderer.goBack() })
            .addAction(textAction("Home") { renderer.load(Prefs.HOME) })
            .addAction(textAction("Search") {
                screenManager.push(SearchScreen(carContext) { query ->
                    renderer.load(Prefs.searchUrl(query))
                })
            })
            .addAction(textAction("Reload") { renderer.reload() })
            .build()

        // PAN lets rotary/joystick head units send scroll events to the surface.
        val mapActions = ActionStrip.Builder().addAction(Action.PAN).build()

        return NavigationTemplate.Builder()
            .setActionStrip(actions)
            .setMapActionStrip(mapActions)
            .build()
    }

    private fun textAction(title: String, onClick: () -> Unit): Action =
        Action.Builder()
            .setTitle(title)
            .setOnClickListener(OnClickListener { onClick() })
            .build()
}
