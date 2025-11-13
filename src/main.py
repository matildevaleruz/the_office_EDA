import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


print("=" * 30)
print("IMPORTACIÓN Y EXPLORACIÓN INICIAL")
print("=" * 30)

# dos datasets de episodios
episodios1 = pd.read_csv("data/the_office_series.csv")
print("=== EPISODIOS1 INFO ===")
episodios1.info()
print("\nPrimeras filas de episodios1:")
print(episodios1.head())
print()

episodios2 = pd.read_csv("data/the_office_imdb.csv")
print("=== EPISODIOS2 INFO ===")
episodios2.info()
print("\nPrimeras filas de episodios2:")
print(episodios2.head())
print()

# comparar columna de votos
print("=== COMPARACIÓN DE VOTOS ===")
print(f"Episodios1 - Total votos: {episodios1.Votes.sum():,}")
print(f"Episodios2 - Total votos: {episodios2.total_votes.sum():,}") # tiene más votos/información más reciente
print()

# dos datasets de líneas de guion
lines1 = pd.read_csv("data/the-office_lines.csv")
print("=== LINES1 INFO ===")
lines1.info()
print("\nPrimeras filas de lines1:")
print(lines1.head())
print()

lines2 = pd.read_csv("data/The-Office-Lines-V4.csv")
print("=== LINES2 INFO ===")
lines2.info() # 4k líneas menos
print("\nPrimeras filas de lines2:")
print(lines2.head())
print()

# explorar los datos de la columna con muchos NaN
print("=== COLUMNA CON NAN EN LINES2 ===")
print(lines2["Unnamed: 6"].value_counts())
print()

# comprobar num de personajes
print("=== COMPARACIÓN DE PERSONAJES ===")
print(f"Lines1 - Personajes únicos: {lines1.Character.nunique()}")
print(f"Lines2 - Personajes únicos: {lines2.speaker.nunique()}")
print()

# comprobar num de valores por personaje
print("=== TOP 10 PERSONAJES POR LÍNEAS ===")
print("Lines1:")
print(lines1.Character.value_counts().reset_index().head(10))
print("\nLines2:")
print(lines2.speaker.value_counts().reset_index().head(10))

print()
print("=" * 30)
print("LIMPIEZA Y TRANSFORMACIÓN")
print("=" * 30)

# eliminar innecesaria
lines = lines1.drop(columns=['Unnamed: 0'])
lines

# estandarizar títulos
lines2['title'] = lines2['title'].str.rstrip(' (Parts 1&2)')

# limpieza y transformación

# al final solo usamos una
episodes = episodios2[['season', 'title', 'imdb_rating']]
print("=== EPISODIOS FILTRADOS ===")
print(episodes)
print()

# eliminar innecesaria
lines = lines1.drop(columns=['Unnamed: 0'])
print("=== LINES LIMPIADO ===")
print(lines)
print()

# estandarizar títulos
lines2['title'] = lines2['title'].str.rstrip(' (Parts 1&2)')

# seleccionar columnas útiles
scenes = lines2[['season', 'episode', 'title', 'scene', 'speaker']]
print("=== ESCENAS FILTRADAS ===")
print(scenes)


# ================
print()
print("=" * 30)
print("DESARROLLO Y ANÁLISIS")
print("=" * 30)

# =========== H1 - episodios ===========
print()
print("=" * 30)
print("H1. Número de episodios")
print("=" * 30)

# seleccionar valores únicos de lines1
df = lines1[['Season', 'Episode_Number', 'Character']].drop_duplicates()

# agrupar por personaje
epXpers = df.groupby("Character")["Episode_Number"].count().sort_values(ascending=False).head(5)
print("\nTOP 5 PERSONAJES CON MÁS EPISODIOS:")
print(epXpers)
print("\n")

# crear gráfico
plt.figure(figsize=(8, 4))
plt.hlines(y=epXpers.index,
           xmin=170,
           xmax=epXpers,
           color='skyblue')
plt.plot(epXpers, epXpers.index, "o")

# añadir count a cada punta
for i, count in enumerate(epXpers):
    plt.text(count + 0.1, i, f'{count}', va='center', ha='left', fontsize=11)

plt.xticks([])

plt.xlabel('Número de episodios', fontsize=12)
plt.ylabel('Personajes', fontsize=12)
plt.title('Número de Episodios con cada Personaje', fontsize=14, fontweight='bold')
plt.tight_layout()

# guardar el gráfico
plt.savefig('episodios_por_personaje.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'episodios_por_personaje.png'")

# ======== H2 - diálogo ===========
print()
print("=" * 30)
print("H1. Cantidad de líneas")
print("=" * 30)

# contar num líneas por personaje
nLin = lines.Character.value_counts().head(10)
print("\n10 PERSONAJES CON MÁS LÍNEAS:")
print(nLin)
print("\n")

# crear gráfico
plt.figure(figsize=(8, 5))

bars = plt.bar(nLin.index, nLin.values)

# Guardar el gráfico
plt.savefig('lineas_por_personaje.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'lineas_por_personaje.png'")

# =========== H3. Ratings ===========
print()
print("=" * 30)
print("H3. Ratings con Dwight")
print("=" * 30)


# número de lineas de Dwight en el episodio
dwight_lines = lines2[lines2['speaker'] == 'Dwight'].groupby(['season', 'title']).size().reset_index(name='dwight_line_count')

# juntar con los ratings
merged_df = pd.merge(dwight_lines, episodes, on=['season', 'title'])

# calcular correlacion
correlation = merged_df['dwight_line_count'].corr(merged_df['imdb_rating'])

print("CORRELACIÓN LÍNEAS DE DWIGHT VS RATING")
print(f"Correlation coefficient: {correlation:.3f}")
print("\n")

# crear scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='dwight_line_count', y='imdb_rating')
sns.regplot(data=merged_df, x='dwight_line_count', y='imdb_rating',
            scatter=False, color='red')

plt.text(75, 8.25, f'r = {correlation:.3f}',
         fontsize=12, color='red', weight='bold')

plt.title('Líneas de Dwight vs Rating IMDb')
plt.xlabel('Número de líneas de Dwight')
plt.ylabel('Rating IMDb')

# Guardar el gráfico
plt.savefig('dwight_lines_vs_rating.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'dwight_lines_vs_rating.png'")


# =========== H4. Escenas compartidas ===========
print()
print("=" * 30)
print("H4. Escenas compartidas")
print("=" * 30)

# eliminar personajes menores (con menos líneas)
top_speakers = scenes.speaker.value_counts().head(20).index
print("\nTOP 20 PERSONAJES MÁS FRECUENTES")
print(top_speakers)
print("\n")

# crear gráfico

# lista de personajes para mostrar en el gráfico
characters = top_speakers

# definir temporadas
seasons = sorted(scenes['season'].unique())

plt.figure(figsize=(12, 6))

# iterar la lista de personajes creando su línea de progreso
for character in characters:
    character_diversity = []
    
    for season in seasons:
        # escenas en las que el personaje aparece
        char_scenes = scenes[
            (scenes['season'] == season) & 
            (scenes['speaker'] == character)
        ]['scene'].unique()
        
        # personajes con los que comparte escena
        co_speakers = scenes[
            (scenes['season'] == season) & 
            (scenes['scene'].isin(char_scenes)) & 
            (scenes['speaker'] != character)
        ]['speaker'].unique()
        
        character_diversity.append(len(co_speakers))
    
    # linea del personaje
    plt.plot(seasons, character_diversity, marker='o', linewidth=2, markersize=6, label=character)

plt.xlabel('Temporadas')
plt.ylabel('Número de personajes con los que interactúa')
plt.title('Número de interacciones con otros personajes')
plt.grid(True, alpha=0.3)
plt.xticks(seasons)
plt.legend()

# Guardar el gráfico
plt.savefig('interacciones_por_personaje.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado como 'interacciones_por_personaje.png'")

