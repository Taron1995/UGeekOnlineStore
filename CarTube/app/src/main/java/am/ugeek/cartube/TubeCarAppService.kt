package am.ugeek.cartube

import android.content.Intent
import androidx.car.app.CarAppService
import androidx.car.app.Screen
import androidx.car.app.Session
import androidx.car.app.validation.HostValidator

class TubeCarAppService : CarAppService() {
    // Personal, sideloaded app: accept any Android Auto host.
    override fun createHostValidator(): HostValidator = HostValidator.ALLOW_ALL_HOSTS_VALIDATOR

    @Suppress("DEPRECATION")
    override fun onCreateSession(): Session = TubeSession()
}

class TubeSession : Session() {
    override fun onCreateScreen(intent: Intent): Screen = TubeScreen(carContext)
}
