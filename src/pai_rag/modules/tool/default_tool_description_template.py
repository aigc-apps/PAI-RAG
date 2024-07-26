DEFAULT_GOOGLE_SEARCH_TOOL_DESP = """
    The google_search tool is a powerful resource for accessing information from the web through the Google search engine. Utilize this tool for retrieving real-time data, conducting research, or finding specific content online, including but not limited to current events, stock market updates, academic articles, and more.

    This tool interfaces directly with Google to provide a streamlined approach to gathering search results tailored to your query. It is ideal for instances where up-to-date and relevant information is required quickly and accurately, such as tracking market trends, understanding breaking news, or accumulating sources for academic purposes.

    google_search(query: str, num: int = None) -> List[str]
    Submit a search query to the Google search engine and receive a list of results. This function allows users to specify the number of search results desired, making it a customizable and flexible tool for various information retrieval needs.

    Args:
        query (str): The search query string that specifies what information to retrieve from Google search.
        num (int, optional): The number of search results to return, giving users the ability to limit the scope of data collected. Defaults to None, which lets Google determine the number of results.

    Raises:
        ValueError: If 'num' is provided and is not an integer between 1 and 10, ensuring user requests are within a manageable range.

    Whether you are looking to gather the latest stock quotes, delve into scholarly literature, or stay informed about global news, the google_search tool stands as a gateway to a vast expanse of information at your fingertips.
"""

DEFAULT_CALCULATE_MULTIPLY = """"
    This tool is designed to assist with a variety of numerical calculations where multiplication is required. It is particularly useful for scenarios such as age computation, financial calculations involving money, quantifying items, and any situation where the product of two integers is sought. The `multiply` function provided by this tool performs integer multiplication and is essential when accuracy and integer results are crucial.

    multiply(a: int, b: int) -> int
    Multiply two integers and returns the result as an integer. This function is ideal for tasks that need to calculate products of numerical values in an accurate and efficient manner.

    Args:
        a (int): The first integer factor in the multiplication.
        b (int): The second integer factor in the multiplication.

    Returns:
        int: The product of multiplying the two integers, suitable for numerical computations that rely on integer values.

    Raises:
        ValueError: If either 'a' or 'b' is not an integer, as non-integer inputs cannot be processed by the multiply function.

    Examples of use include multiplying quantities of items in inventory management, calculating the total cost from unit price and quantity in financial transactions, computing square footage, and many other practical applications where multiplication of integers is necessary.
"""

DEFAULT_CALCULATE_ADD = """
    The calculate_add tool provides a reliable way to perform addition operations for a wide range of numerical computing needs. It is an essential utility for tasks that require summing of integer values, such as tallying scores, aggregating data, calculating financial totals, or even determining cumulative age. The `add` function within this tool strictly handles addition of two integers, ensuring precise and integer-specific computation.

    add(a: int, b: int) -> int
    Add two integers and return the result as an integer. This function is particularly useful for straightforward arithmetic operations where the total sum of two numbers is needed without the complexity of handling floats or decimals.

    Args:
        a (int): The first integer to be added.
        b (int): The second integer to be added.

    Returns:
        int: The sum of the two integers, ideal for use in contexts demanding accurate arithmetic operations involving integer values.

    Raises:
        ValueError: If either 'a' or 'b' is not an integer, since the add function is tailored to handle integer addition only.

    Example scenarios where this tool can be applied include but are not limited to adding up expenses, combining quantities of different items in stock, computing the total number of days between dates for planning purposes, and various other applications where adding integers is crucial.
"""

DEFAULT_CALCULATE_DIVIDE = """
    The calculate_divide tool is indispensable for performing division operations in various numerical contexts that require precise quotient determination. It holds particular significance for calculating ratios, determining average values, assessing financial rates, partitioning quantities, and other scenarios where division of integers produces a floating-point result.

    divide(a: int, b: int) -> float
    Divide one integer by another and return the quotient as a float. This function excels in cases where division might result in a non-integer outcome, ensuring accuracy and detail by retaining the decimal part of the quotient.

    Args:
        a (int): The numerator, or the integer to be divided.
        b (int): The denominator, or the integer by which to divide.

    Returns:
        float: The floating-point result of the division, which is suitable for computations that demand more precision than integer division can provide.

    Raises:
        ValueError: If 'b' is zero since division by zero is undefined, or if either 'a' or 'b' is not an integer.

    Practical applications for this tool are widespread: it can aid in financial computations like determining price per unit, in educational settings for calculating grade point averages, or in any sector where division is a fundamental operation and its exact floating-point result is needed.
"""

DEFAULT_CALCULATE_SUBTRACT = """
    The calculate_subtract tool is designed to facilitate subtraction operations in a variety of numerical calculations. It is an invaluable resource for determining the difference between two integer values, such as calculating the remaining balance, evaluating data discrepancies, computing change in financial transactions, or quantifying the decrease in stock levels.

    subtract(a: int, b: int) -> int
    Subtract the second integer from the first and return the difference as an integer. This function is crucial for operations that require an exact integer result, avoiding the potential rounding errors associated with floating-point arithmetic.

    Args:
        a (int): The minuend, or the integer from which the second integer is to be subtracted.
        b (int): The subtrahend, or the integer to be subtracted from the first integer.

    Returns:
        int: The integer result representing the difference between the two integers, perfect for situations where integral precision is needed in subtraction.

    Raises:
        ValueError: If either 'a' or 'b' is not an integer, as the subtract function is strictly for integer arithmetic.

    Example uses of this tool include but are not limited to calculating age differences, determining the number of items sold from inventory, working out loan repayments, and any other context where subtraction of numerical values plays a key role.
"""

DEFAULT_GET_WEATHER = """
    The get_weather tool has been meticulously crafted to fetch real-time weather data for any global location, empowering users with accurate meteorological insights. Whether you're planning outdoor activities, assessing travel conditions, monitoring agricultural climates, or simply staying informed about the day's weather, this tool offers a streamlined solution. It taps into reputable weather APIs to deliver up-to-date information on temperature, humidity, precipitation, wind conditions, and more, ensuring you're equipped with the latest atmospheric conditions.

    get_weather(city: str) -> str
    This function not only provides current weather conditions but also encompasses forecast data when supported by the API, thereby catering to a wide array of weather-dependent decision-making processes. The returned dictionary encapsulates various weather parameters, enabling detailed analysis tailored to your needs.

    Args:
        city (str): The name of the city, town, or specific location for which weather data is desired. Ensure the input adheres to the API's naming conventions for optimal results.

    Returns:
        str: A neatly packaged string presenting the time, city, temperature in Celsius or Fahrenheit (as per the API's standard setting), and a brief description of the weather, such as "sunny," "partly cloudy," or "light rain." This format facilitates easy reading and can be seamlessly integrated into messages, notifications, or displayed on-screen.

    Raises:
        ValueError: If the 'city' input is invalid or unrecognizable, ensuring that errors are promptly communicated for corrective action.

    Embracing versatility, the get_weather tool finds application in travel planning, event organization, health advisories related to extreme weather, educational projects studying climatology, and everyday life decisions influenced by the elements. Its capability to distill complex meteorological data into digestible insights underscores its value as an indispensable utility in understanding our dynamic atmospheric surroundings.
"""
