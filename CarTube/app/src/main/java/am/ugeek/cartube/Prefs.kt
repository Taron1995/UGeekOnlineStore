package am.ugeek.cartube

import android.content.Context
import java.net.URLEncoder

/** Shared state between the phone screen and the car screen. */
object Prefs {
    const val NAME = "cartube"
    const val KEY_LAST_URL = "last_url"
    const val KEY_PENDING_URL = "pending_url"
    const val KEY_PENDING_NONCE = "pending_nonce"
    const val HOME = "https://m.youtube.com/"

    fun searchUrl(query: String): String =
        "https://m.youtube.com/results?search_query=" + URLEncoder.encode(query, "UTF-8")

    /** Text typed on the phone: a link is opened as-is, anything else is searched. */
    fun urlFor(input: String): String {
        val text = input.trim()
        return when {
            text.isEmpty() -> HOME
            text.startsWith("http://") || text.startsWith("https://") -> text
            else -> searchUrl(text)
        }
    }

    /** Ask the car screen to open [url]. The nonce makes repeated sends of the same URL fire. */
    fun sendToCar(context: Context, url: String) {
        context.getSharedPreferences(NAME, Context.MODE_PRIVATE).edit()
            .putString(KEY_PENDING_URL, url)
            .putString(KEY_LAST_URL, url)
            .putLong(KEY_PENDING_NONCE, System.nanoTime())
            .apply()
    }
}
