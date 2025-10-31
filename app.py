import asyncio
import logging
import os
from contextlib import suppress
from typing import Dict, Any, Optional

from dotenv import load_dotenv  # Load .env otomatis
load_dotenv()

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
from aiogram.enums.parse_mode import ParseMode
from aiogram.client.default import DefaultBotProperties  # default bot props (aiogram >= 3.7)

from cfg import cfg
from core.process import process_image_pipeline
from core.utils import parse_target_number, ensure_dirs
from core.check import verify_template_pack

logging.basicConfig(level=logging.INFO)
router = Router()

USER_STATE: Dict[int, Dict[str, Any]] = {}
GLOBAL_SEMAPHORE = asyncio.Semaphore(cfg["max_global_workers"])

def get_user_state(uid: int) -> Dict[str, Any]:
    if uid not in USER_STATE:
        USER_STATE[uid] = {
            "mode": "android",
            "queue": asyncio.Queue(maxsize=cfg["per_user_queue"]),
            "worker": None
        }
    return USER_STATE[uid]

async def user_worker(bot: Bot, uid: int):
    state = get_user_state(uid)
    queue = state["queue"]
    processing_msg: Optional[Message] = None
    try:
        while True:
            item = await queue.get()
            try:
                if processing_msg is None:
                    processing_msg = await bot.send_message(
                        chat_id=uid,
                        text="⏳ Memproses screenshot... mohon tunggu sebentar."
                    )

                async with GLOBAL_SEMAPHORE:
                    result_path, meta = await process_image_pipeline(
                        file_path=item["file_path"],
                        mode=item["mode"],
                        user_caption=item["caption"],
                        user_id=uid
                    )

                caption = f"Selesai ✅\nMode: {meta['mode']} • Tema: {meta['theme']}"
                # Kirim sebagai foto (compressed) agar langsung terlihat di chat
                await bot.send_photo(
                    chat_id=uid,
                    photo=FSInputFile(result_path),
                    caption=caption
                )
            except Exception as e:
                logging.exception("Processing failed")
                await bot.send_message(uid, f"⚠️ Gagal memproses gambar: {e}")
            finally:
                queue.task_done()

            if queue.empty() and processing_msg:
                with suppress(Exception):
                    await processing_msg.delete()
                processing_msg = None
    except asyncio.CancelledError:
        pass

@router.message(Command("start"))
async def cmd_start(msg: Message):
    get_user_state(msg.from_user.id)
    await msg.answer(
        "Halo! 👋 Kirim screenshot Info Grup WhatsApp + caption angka target.\n"
        "Pilih mode:\n"
        "/android • /iphone • /all (deteksi otomatis)\n\n"
        "Contoh: setelah /android, kirim foto dengan caption 1234"
    )

@router.message(Command("help"))
async def cmd_help(msg: Message):
    await msg.answer(
        "Panduan singkat:\n"
        "1) /android atau /iphone untuk set mode manual, /all untuk auto.\n"
        "2) Kirim screenshot dengan caption angka target (contoh: 2578).\n"
        "3) Maks 5 gambar di antrian per user. Proses paralel & aman.\n"
        "Catatan: Bot mengirim hasil sebagai foto (compressed) agar langsung terlihat."
    )

@router.message(Command("android"))
async def cmd_android(msg: Message):
    st = get_user_state(msg.from_user.id)
    st["mode"] = "android"
    await msg.answer("Mode disetel ke ANDROID ✅\nKirim screenshot + caption angka target.")

@router.message(Command("iphone"))
async def cmd_iphone(msg: Message):
    st = get_user_state(msg.from_user.id)
    st["mode"] = "iphone"
    await msg.answer("Mode disetel ke IPHONE ✅\nKirim screenshot + caption angka target.")

@router.message(Command("all"))
async def cmd_all(msg: Message):
    st = get_user_state(msg.from_user.id)
    st["mode"] = "all"
    await msg.answer("Mode disetel ke AUTO-DETECT ✅\nKirim screenshot + caption angka target.")

@router.message(F.photo | F.document)
async def handle_image(msg: Message, bot: Bot):
    st = get_user_state(msg.from_user.id)
    mode = st["mode"]

    caption = (msg.caption or "").strip()
    target_number = parse_target_number(caption)
    if target_number is None:
        await msg.answer("Mohon sertakan caption angka target. Contoh: 1234")
        return

    file_id = msg.photo[-1].file_id if msg.photo else msg.document.file_id
    file = await bot.get_file(file_id)
    ensure_dirs(cfg["work_dir"])
    local_path = os.path.join(cfg["work_dir"], f"{msg.from_user.id}_{file.file_unique_id}.bin")
    await bot.download_file(file.file_path, destination=local_path)

    queue = st["queue"]
    if queue.full():
        await msg.answer("Antrian Anda sudah 5 pekerjaan. Mohon tunggu hasil sebelumnya ya 🙏")
        return

    await queue.put({
        "file_path": local_path,
        "caption": caption,
        "mode": mode
    })

    if not st["worker"] or st["worker"].done():
        st["worker"] = asyncio.create_task(user_worker(bot, msg.from_user.id))

    pos = queue.qsize()
    await msg.answer(f"✅ Ditambahkan ke antrian. Posisi Anda: {pos}. Akan diproses segera.")

async def startup_template_check():
    missing = []
    for mode in ("android", "iphone"):
        for theme in ("light", "dark"):
            errs = verify_template_pack(mode, theme)
            if errs:
                missing.extend(errs)
    if missing:
        logging.warning("Beberapa template belum lengkap:\n- " + "\n- ".join(missing))
    else:
        logging.info("Semua template minimal tersedia.")

async def main():
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN belum diisi. Lihat .env.example")
    await startup_template_check()

    # aiogram >= 3.7: gunakan DefaultBotProperties untuk parse_mode default
    bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
