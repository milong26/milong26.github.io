import { Plugin, TFile } from "obsidian";

export default class AutoTemplatePlugin extends Plugin {

	async onload() {
		this.registerEvent(
			this.app.vault.on("create", async (file) => {
				if (!(file instanceof TFile)) return;
				if (file.extension !== "md") return;

				const content = await this.app.vault.read(file);
				if (content.trim().length > 0) return;

				const tplFile = this.app.vault.getAbstractFileByPath("template/log.md");
				if (!(tplFile instanceof TFile)) return;

				const template = await this.app.vault.read(tplFile);
				await this.app.vault.modify(file, template);
			})
		);
	}
}