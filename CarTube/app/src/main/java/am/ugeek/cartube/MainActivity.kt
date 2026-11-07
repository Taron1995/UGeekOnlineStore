package am.ugeek.cartube

import android.app.Activity
import android.os.Bundle
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast

/**
 * Phone screen: type a search or paste a YouTube link and send it to the car.
 * Typing on the phone is easier than the car keyboard, which is often locked.
 */
class MainActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()

        val input = EditText(this).apply {
            hint = "Որոնում կամ YouTube հղում"
            setSingleLine()
            imeOptions = EditorInfo.IME_ACTION_SEND
        }

        fun send(url: String) {
            Prefs.sendToCar(this, url)
            Toast.makeText(this, "Ուղարկվեց մեքենային", Toast.LENGTH_SHORT).show()
        }

        input.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                send(Prefs.urlFor(input.text.toString())); true
            } else false
        }

        val sendButton = Button(this).apply {
            text = "Ուղարկել մեքենային"
            setOnClickListener { send(Prefs.urlFor(input.text.toString())) }
        }
        val homeButton = Button(this).apply {
            text = "YouTube գլխավոր էջ"
            setOnClickListener { send(Prefs.HOME) }
        }

        val help = TextView(this).apply {
            textSize = 15f
            text = """
                CarTube — YouTube Android Auto-ում

                1. Android Auto → Version-ի վրա սեղմեք 10 անգամ → ⋮ → Developer settings → միացրեք «Unknown sources»։
                2. Միացրեք հեռախոսը մեքենային, Android Auto-ի ցանկում բացեք CarTube-ը։
                3. Այստեղ գրեք որոնում կամ հղում և սեղմեք «Ուղարկել մեքենային»։

                Redmi/Xiaomi: Settings → Apps → CarTube և Android Auto → Battery saver → No restrictions, միացրեք Autostart։

                ⚠ Տեսանյութ դիտեք միայն կայանված ժամանակ։
            """.trimIndent()
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            addView(input)
            addView(sendButton)
            addView(homeButton)
            addView(help)
        }
        setContentView(ScrollView(this).apply { addView(layout) })
    }
}
