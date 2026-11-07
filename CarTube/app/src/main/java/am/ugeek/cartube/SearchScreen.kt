package am.ugeek.cartube

import androidx.car.app.CarContext
import androidx.car.app.Screen
import androidx.car.app.model.Action
import androidx.car.app.model.ItemList
import androidx.car.app.model.SearchTemplate
import androidx.car.app.model.Template

/** On-screen car keyboard for a YouTube search (the car may block it while driving). */
class SearchScreen(
    carContext: CarContext,
    private val onSearch: (String) -> Unit,
) : Screen(carContext) {

    override fun onGetTemplate(): Template =
        SearchTemplate.Builder(object : SearchTemplate.SearchCallback {
            override fun onSearchSubmitted(searchText: String) {
                if (searchText.isNotBlank()) onSearch(searchText)
                screenManager.pop()
            }
        })
            .setHeaderAction(Action.BACK)
            .setSearchHint("YouTube")
            .setShowKeyboardByDefault(true)
            .setItemList(ItemList.Builder().setNoItemsMessage("Type and press search").build())
            .build()
}
