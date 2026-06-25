const { Plugin, TFile } = require("obsidian");

module.exports = class AutoTemplatePlugin extends Plugin {

	async onload() {

		this.registerEvent(
			this.app.vault.on("create", async (file) => {

				if (!(file instanceof TFile)) return;
				if (file.extension !== "md") return;

				// 读取新文件内容
				const content = await this.app.vault.read(file);

				// 已有内容则跳过
				if (content && content.trim().length > 0) return;

				// 读取模板
				const tplPath = "template/log.md";
				const tplFile = this.app.vault.getAbstractFileByPath(tplPath);

				if (!(tplFile instanceof TFile)) return;

				const template = await this.app.vault.read(tplFile);

				// 写入模板
				await this.app.vault.modify(file, template);
			})
		);
	}
};