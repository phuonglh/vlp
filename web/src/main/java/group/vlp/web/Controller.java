package group.vlp.web;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Predicate;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

/**
 * Controller for beans and entities. 
 * <p/>
 * @author phuonglh
 *
 */
@RestController
public class Controller {
	static List<Paper> papers = new LinkedList<>();
	static List<Book> books = new LinkedList<>();
	
	static {
		// 0. papers
		Paper j1 = new Journal(new String[]{"Phuong Le-Hong", "Erik Cambria"}, 
				"A Semantics-Aware Approach for Multilingual Natural Language Inference", 
				"Language Resources and Evaluation",
				2023, true, "Springer Nature", 57, 2, "611-649");
		j1.setKeywords(Arrays.asList(new String[] {"lrev", "semantics", "nli", "inference"}));
		papers.add(j1);
		Paper j2 = new Journal(new String[]{"Chi Tho Luong", "Phuong Le-Hong", "Thi Oanh Tran"}, 
				"A rich task-oriented dialogue corpus in Vietnamese", 
				"Language Resources and Evaluation",
				2022, false, "Springer Nature", 0, 0, "");
		j2.setKeywords(Arrays.asList(new String[] {"lrev", "dialogue", "corpus"}));
		papers.add(j2);
		Paper c1 = new Conference(new String[]{"Phuong Le-Hong", "et al."}, 
				"Multilingual Natural Language Understanding for the FPT.AI Conversational Platform",
				"Proceedings of the 14th IEEE International Conference on Knowledge and Systems Engineering",
				2022, true, "IEEE", "Nha Trang, Vietnam");
		c1.setKeywords(Arrays.asList(new String[] {"kse", "nlu", "platform", "fpt"}));
		papers.add(c1);
		
		// 2. books
		Book b1 = new Book("Tim Marshal", "Phan Linh Lan", "Prisoners of Geography", 2020);
		books.add(b1);
		Book b2 = new Book("Tim Marshal", "Trần Trọng Hải Minh", "Divided - Why We're Living in an Age of Walls", 2021);
		books.add(b2);
	}
	
	/**
	 * Lists all papers in the database.
	 * @return a list of papers (journals or conferences).
	 */
	@GetMapping(path = "/papers")
	public List<Paper> getPapers() {
		return papers;
	}
	
	/**
	 * Finds papers with a given keyword in the request path.
	 * @param keyword a given keyword
	 * @return a list of papers
	 */
	@GetMapping(path = "/papers/{keyword}")
	public List<Paper> getPapers(@PathVariable String keyword) {
		Predicate<? super Paper> predicate = paper -> paper.getKeywords().contains(keyword);
		return papers.stream().filter(predicate).toList();
	}
	
	@GetMapping(path = "/books")
	public List<Book> getBooks() {
		return books;
	}
}
