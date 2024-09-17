query_1 = "What are the benefits of renewable energy?"  # gt is [0, 3]
query_2 = "How do solar panels impact the environment?"  # gt is [1, 2]

documents = [
    {
        "title": "The Impact of Renewable Energy on the Economy",
        "content": "Renewable energy technologies not only help in reducing greenhouse gas emissions but also contribute significantly to the economy by creating jobs in the manufacturing and installation sectors. The growth in renewable energy usage boosts local economies through increased investment in technology and infrastructure.",
    },
    {
        "title": "Understanding Solar Panels",
        "content": "Solar panels convert sunlight into electricity by allowing photons, or light particles, to knock electrons free from atoms, generating a flow of electricity. Solar panels are a type of renewable energy technology that has been found to have a significant positive effect on the environment by reducing the reliance on fossil fuels.",
    },
    {
        "title": "Pros and Cons of Solar Energy",
        "content": "While solar energy offers substantial environmental benefits, such as reducing carbon footprints and pollution, it also has downsides. The production of solar panels can lead to hazardous waste, and large solar farms require significant land, which can disrupt local ecosystems.",
    },
    {
        "title": "Renewable Energy and Its Effects",
        "content": "Renewable energy sources like wind, solar, and hydro power play a crucial role in combating climate change. They do not produce greenhouse gases during operation, making them essential for sustainable development. However, the initial setup and material sourcing for these technologies can still have environmental impacts.",
    },
]
