#lang racket
(define (solve s)
  (define has-letter (for/or ([c (in-string s)]) (char-alphabetic? c)))
  (if has-letter
      (list->string (map (lambda (c)
                           (cond [(char-upper-case? c) (char-downcase c)]
                                 [(char-lower-case? c) (char-upcase c)]
                                 [else c])) (string->list s)))
      (list->string (reverse (string->list s)))))
