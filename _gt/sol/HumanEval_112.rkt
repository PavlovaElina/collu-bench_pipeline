#lang racket
(define (reverse_delete s c)
  (define cset (string->list c))
  (define res (list->string (filter (lambda (ch) (not (member ch cset))) (string->list s))))
  (list res (string=? res (list->string (reverse (string->list res))))))
