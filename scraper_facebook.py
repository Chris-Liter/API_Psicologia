from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue as ThreadQueue
from playwright.sync_api import sync_playwright
import json
import time
import os

def scrape_facebook(frase_busqueda, queue):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state="estado_facebook.json")
        page = context.new_page()
        publicaciones = []

        page.goto('https://www.facebook.com/')
        time.sleep(5)

        page.type('input[placeholder="Buscar en Facebook"]', frase_busqueda, delay=120)
        page.keyboard.press('Enter')
        time.sleep(7)

        try:
            publicaciones_btn = page.locator('//span[text()="Publicaciones"]').first
            publicaciones_btn.wait_for(state='visible', timeout=10000)
            publicaciones_btn.click()
            time.sleep(3)

            recientes_btn = page.locator('input[aria-label="Publicaciones recientes"]')
            recientes_btn.wait_for(state='visible', timeout=5000)
            recientes_btn.click()
            time.sleep(2)
        except Exception as e:
            print("⚠️ Error con botones de navegación:", e)

        for _ in range(2):
            page.evaluate("window.scrollBy(0, window.innerHeight)")
            time.sleep(2)

        posts = page.query_selector_all('div.x1yztbdb.x1n2onr6.xh8yej3.x1ja2u2z')
        processed = set()

        for post in posts:
            try:
                if post.is_visible():
                    post_html = post.inner_html()
                    if post_html in processed:
                        continue
                    processed.add(post_html)

                    comment_button = post.query_selector('[data-ad-rendering-role="comment_button"]')
                    if not comment_button:
                        continue

                    try:
                        with context.expect_page(timeout=5000) as new_page:
                            comment_button.click()
                        popup = new_page.value
                        popup.wait_for_load_state('load')
                        target_page = popup
                    except:
                        target_page = page
                        time.sleep(5)

                    titulo = 'Sin título'
                    try:
                        story = post.query_selector('div[data-ad-rendering-role="story_message"]')
                        if story:
                            textos = story.query_selector_all('div[dir="auto"][style*="text-align: start;"]')
                            texto_list = [t.text_content().strip() for t in textos if t.text_content().strip()]
                            if texto_list:
                                titulo = ' '.join(texto_list).replace('\n', ' ').strip()
                    except:
                        pass

                    for _ in range(5):
                        boton = target_page.query_selector('text="Ver más comentarios"')
                        if boton:
                            try:
                                boton.click()
                                time.sleep(2)
                            except:
                                break

                    comentarios_extraidos = []
                    contenedores = target_page.query_selector_all('div.x18xomjl.xbcz3fp')
                    for cont in contenedores:
                        comentarios = cont.query_selector_all(
                            'div.x1lliihq.xjkvuk6.x1iorvi4 div.xdj266r.x14z9mp.xat24cr.x1lziwak.x1vvkbs'
                        )
                        for comentario in comentarios:
                            texto = comentario.text_content().strip()
                            if texto:
                                comentarios_extraidos.append(texto)

                    publicaciones.append({
                        "titulo": titulo,
                        "comentarios": comentarios_extraidos,
                        "frase": frase_busqueda
                    })

                    if target_page != page:
                        target_page.close()
                    else:
                        page.keyboard.press("Escape")
                        time.sleep(2)

                    time.sleep(3)
            except:
                pass

        context.close()
        browser.close()
        print(f"✅ Scraping terminado para frase: {frase_busqueda}")
        queue.put(publicaciones)


# 🔧 Nueva función con threading para evitar bloqueos
def scrape_personas_por_ciudad(frase_busqueda, ciudad, queue):
    resultados = []
    vistos = set()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(storage_state="estado_facebook.json")
            page = context.new_page()

            page.goto("https://www.facebook.com/")
            time.sleep(5)

            page.type('input[placeholder="Buscar en Facebook"]', frase_busqueda, delay=80)
            page.keyboard.press('Enter')
            time.sleep(5)

            try:
                personas_tab = page.locator('//span[text()="Personas"]').first
                personas_tab.wait_for(state='visible', timeout=10000)
                personas_tab.click()
                time.sleep(3)
            except Exception as e:
                print(f"⚠️ No se pudo hacer clic en 'Personas': {e}")
                queue.put(resultados)
                return

            try:
                input_ciudad = page.locator('input[aria-label="Ciudad"]')
                input_ciudad.wait_for(state='visible', timeout=10000)
                input_ciudad.fill(ciudad)
                time.sleep(2)

                sugerencia = page.locator('ul[role="listbox"] li').first
                sugerencia.wait_for(state='visible', timeout=10000)
                sugerencia.click()
                time.sleep(4)
            except Exception as e:
                print(f"⚠️ No se pudo aplicar el filtro de ciudad '{ciudad}': {e}")

            cards = page.query_selector_all('div.x78zum5.xdt5ytf')

            for card in cards:
                try:
                    enlace = card.query_selector('a[href^="https://www.facebook.com/"]')
                    if enlace:
                        nombre = enlace.text_content().strip()
                        perfil = enlace.get_attribute('href')

                        # Evitar duplicados
                        key = (nombre, perfil)
                        if key in vistos:
                            continue
                        vistos.add(key)

                        resultados.append({
                            "nombre": nombre,
                            "perfil": perfil,
                            "frase": frase_busqueda,
                            "ciudad": ciudad
                        })
                except:
                    continue

            context.close()
            browser.close()
            print(f"✅ Scraping de personas terminado para ciudad: {ciudad}")

    except Exception as e:
        print(f" Error general en scraping de personas por ciudad: {e}")
    finally:
        queue.put(resultados)



def scrape_usuario_especifico(username, queue):
    publicaciones = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(storage_state="estado_facebook.json")
            page = context.new_page()

            url_perfil = f"https://www.facebook.com/{username}"
            page.goto(url_perfil)
            time.sleep(6)

            # Scroll hacia abajo dos veces para cargar más publicaciones
            for _ in range(2):
                page.mouse.wheel(0, 2500)
                time.sleep(2)

            # Volver al inicio
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(2)

            posts = page.query_selector_all('div.x1yztbdb.x1n2onr6.xh8yej3.x1ja2u2z')
            print(f" Encontradas {len(posts)} publicaciones")
            vistos = set()

            for idx, post in enumerate(posts):
                try:
                    if not post.is_visible():
                        continue

                    post_html = post.inner_html()
                    if post_html in vistos:
                        continue
                    vistos.add(post_html)

                    # Scroll al post completo por si no es visible aún
                    try:
                        post.scroll_into_view_if_needed()
                        time.sleep(1)
                    except:
                        pass

                    # Buscar botón de comentarios
                    boton_comentar = post.query_selector('div[role="button"][aria-label*="comentario"]')
                    if not boton_comentar:
                        print(f"⏩ Publicación {idx + 1} sin botón comentar visible, se omite.")
                        continue

                    if not boton_comentar.is_visible():
                        print(f"⏩ Botón comentar no visible en publicación {idx + 1}, se omite.")
                        continue

                    try:
                        boton_comentar.evaluate("el => el.click()")
                        time.sleep(3)
                        print(f"✅ Comentarios abiertos en publicación {idx + 1}")

                        #  Scroll en el panel de comentarios específico
                        try:
                            page.evaluate("""
                                () => {
                                    const panel = document.querySelector('div.xb57i2i.x1q594ok.x5lxg6s.x78zum5.xdt5ytf.x6ikm8r.x1ja2u2z.x1pq812k.x1rohswg.xfk6m8.x1yqm8si.xjx87ck.xx8ngbg.xwo3gff.x1n2onr6.x1oyok0e.x1odjw0f.x1iyjqo2.xy5w88m');
                                    if (panel) {
                                        panel.scrollBy(0, 2000);
                                    }
                                }
                            """)
                            print("🔄 Scroll realizado en el panel de comentarios")
                            time.sleep(3)  # Esperar a que se carguen nuevos comentarios
                        except Exception as e:
                            print(f"⚠️ No se pudo hacer scroll: {e}")


                    except Exception as e:
                        print(f"⚠️ Error al hacer clic en comentario de publicación {idx + 1}: {e}")
                        continue


                    
                    #  Título (después del clic)
                    titulo = "Sin contenido"
                    try:
                        texto_div = post.query_selector('div[dir="auto"][style*="text-align: start;"]')
                        if texto_div:
                            titulo = texto_div.inner_text().strip()
                            if not titulo:
                                span = texto_div.query_selector('span')
                                if span:
                                    titulo = span.inner_text().strip()
                    except:
                        pass

                    #  Comentarios
                    comentarios_extraidos = []

                    # 1. Buscar todos los contenedores de comentarios
                    bloques_comentarios = page.query_selector_all('div.x78zum5.xdt5ytf[data-virtualized="false"]')


                    for bloque in bloques_comentarios:
                        try:
                            # 2. Dentro de cada uno, buscar el texto real del comentario
                            comentario_contenedor = bloque.query_selector('div.xdj266r.x14z9mp.xat24cr.x1lziwak.x1vvkbs')
                            if comentario_contenedor:
                                texto_div = comentario_contenedor.query_selector('div[dir="auto"][style*="text-align: start;"]')
                                if texto_div:
                                    texto = texto_div.inner_text().strip()
                                    if not texto:
                                        span = texto_div.query_selector('span')
                                        if span:
                                            texto = span.inner_text().strip()
                                    if texto and texto != titulo and texto not in comentarios_extraidos:
                                        comentarios_extraidos.append(texto)
                        except Exception as e:
                            print(f"⚠️ Error extrayendo un comentario: {e}")
                            continue

                    publicaciones.append({
                        "usuario": username,
                        "titulo": titulo,
                        "comentarios": comentarios_extraidos
                    })

                    # Salir del comentario
                    page.keyboard.press("Escape")
                    time.sleep(2)

                except Exception as e:
                    print(f"❌ Error al procesar publicación {idx + 1}: {e}")
                    continue

            context.close()
            browser.close()
            print(f"✅ Scraping terminado para usuario: {username}")

    except Exception as e:
        print(f" Error general scraping usuario: {e}")

    finally:
        queue.put(publicaciones)








def run_parallel_scraping():
    q1 = Queue()
    q2 = Queue()
    q3 = ThreadQueue()
    q4 = ThreadQueue()

    # Leer las frases desde archivo generado por app.py
    if os.path.exists("frases_scraping.json"):
        with open("frases_scraping.json", "r", encoding="utf-8") as f:
            frases = json.load(f)
    else:
        print(" No se encontró frases_scraping.json")
        frases = ["placeholder 1", "placeholder 2"]

    # Leer ciudad desde archivo externo generado por el backend
    if os.path.exists("ciudad_scraping.txt"):
        with open("ciudad_scraping.txt", "r", encoding="utf-8") as f:
            ciudad = f.read().strip()
    else:
        ciudad = "Cuenca"

    
    # Lee usuario objetivo desde archivo
    if os.path.exists("usuario_scraping.txt"):
        with open("usuario_scraping.txt", "r", encoding="utf-8") as f:
            usuario_objetivo = f.read().strip()
    else:
        usuario_objetivo = "zuck"  # Por defecto, Mark Zuckerberg

    p1 = Process(target=scrape_facebook, args=(frases[0], q1))
    p2 = Process(target=scrape_facebook, args=(frases[1], q2))
    t3 = Thread(target=scrape_personas_por_ciudad, args=("Psicologos", ciudad, q3))
    t4 = Thread(target=scrape_usuario_especifico, args=(usuario_objetivo, q4))

    p1.start()
    p2.start()
    t3.start()
    t4.start()

    p1.join()
    print("▶ p1.join() completado")
    p2.join()
    print("▶ p2.join() completado")
    t3.join()
    print("▶ t3.join() completado")
    t4.join()
    print("▶ t4.join() completado")

    print(f" Proceso 1 terminó con exit code: {p1.exitcode}")
    print(f" Proceso 2 terminó con exit code: {p2.exitcode}")

    resultados = []
    personas = []
    publicaciones_usuario = []

    try:
        resultados += q1.get(timeout=30)
        resultados += q2.get(timeout=30)
        personas += q3.get(timeout=30)
        publicaciones_usuario += q4.get(timeout=30)
    except Exception as e:
        print(" Error al obtener resultados:", e)

    with open("comentariosFacebookMultiprocesoFinal.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    with open("personasFacebookCiudad.json", "w", encoding="utf-8") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)
    
    with open("publicacionesUsuario.json", "w", encoding="utf-8") as f:
        json.dump(publicaciones_usuario, f, indent=2, ensure_ascii=False)

    print("✅ Archivos guardados correctamente.")
    import sys
    sys.exit(0)



if __name__ == "__main__":
    run_parallel_scraping()
